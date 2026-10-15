"""Execution-based "smoke test" scoring tier -- a cheaper alternative to full
execution-based scoring. Actually calls model-generated code (unlike every
scorer in scorers.py, which is purely AST/text-based) to catch behavior
changes plain structural comparison is blind to -- most notably a return-type
change (gold's sum() -> a scalar vs. pred's raw boolean-filtered DataFrame, a
type mismatch invisible to node-count-ratio scoring).

SAFETY: this corpus's snippets contain real sqlite3/requests/os/subprocess
calls as "realistic" example content. Every exec here
runs inside a sandboxed namespace:
  - a custom __import__ hook whitelists a short list of safe, deterministic,
    no-I/O modules (see _SAFE_MODULES) and returns a MagicMock for anything
    else, so e.g. `sqlite3.connect(x).cursor().execute("DROP TABLE users")`
    silently becomes a harmless mock-chain call -- no real file, no real
    query, ever.
  - dangerous builtins (open/exec/eval/compile/input/__import__) are
    replaced with blocked/inert stand-ins.
  - each call runs in a worker thread with a hard timeout, so a runaway
    loop can't hang the caller. Python has no safe way to forcibly kill a
    thread, so a timed-out thread is abandoned (daemon=True keeps it from
    blocking process exit) rather than actually stopped -- an accepted
    limitation for a best-effort heuristic tier, not used in a tight
    unbounded loop against untrusted input at scale.

Deliberately NOT wired into scorers.py's TASK_SCORERS / OK_SCORE_THRESHOLD --
opt-in only (SMOKE_TEST_SCORERS below), same discipline TASK_SCORERS_SEMANTIC
already follows: never silently changes the score an in-flight eval or RL
run sees.
"""

import ast as _ast
import builtins as _builtins_mod
import contextlib as _contextlib
import inspect as _inspect
import io as _io
import threading as _threading
from unittest.mock import MagicMock

from .scorers import _func_names

DEFAULT_TIMEOUT_S = 2.0

# Modules real enough to be worth actually importing (deterministic,
# in-memory, no filesystem/network/process/env access) -- everything else
# gets a MagicMock instead. pandas/numpy are common in this corpus's
# "realistic" examples and do real, deterministic, no-I/O computation, so
# they're allowed IF actually installed; never required.
_SAFE_MODULES = {
    "math", "json", "re", "itertools", "functools", "collections",
    "datetime", "string", "copy", "statistics", "decimal", "fractions",
    "typing", "dataclasses", "enum", "textwrap", "random", "operator",
    "heapq", "bisect", "array",
}
for _opt in ("pandas", "numpy"):
    try:
        __import__(_opt)
        _SAFE_MODULES.add(_opt)
    except ImportError:
        pass

_BLOCKED_BUILTIN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}


def _blocked_open(*args, **kwargs):
    raise PermissionError("file I/O is disabled in the smoke-test sandbox")


def _make_sandboxed_import():
    real_import = _builtins_mod.__import__

    def sandboxed_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top in _SAFE_MODULES:
            return real_import(name, globals, locals, fromlist, level)
        # Stand-in for the whole module: attribute access, calls, and
        # chained calls all silently succeed and return further Mocks, so
        # `sqlite3.connect(x).cursor().execute("DROP TABLE users")` or
        # `requests.get(url)` never touch a real database/network/filesystem.
        return MagicMock(name=f"sandboxed_module:{name}")

    return sandboxed_import


def _sandbox_builtins():
    safe = {k: v for k, v in vars(_builtins_mod).items() if k not in _BLOCKED_BUILTIN_NAMES}
    safe["__import__"] = _make_sandboxed_import()
    safe["open"] = _blocked_open
    safe["input"] = lambda *a, **k: ""
    return safe


def _run_with_timeout(fn, timeout_s=DEFAULT_TIMEOUT_S):
    """Runs fn() in a worker thread. Returns (completed, result_or_exception).
    completed=False on timeout OR on any exception fn() raised -- callers
    distinguish the two via isinstance(result, TimeoutError) if it matters,
    but for this module's purposes both just mean "couldn't get a clean
    result," treated the same way (smoke_ok can't be established)."""
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001 -- deliberately broad, this IS the safety boundary
            result["error"] = e

    t = _threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return False, TimeoutError(f"smoke test exceeded {timeout_s}s")
    if "error" in result:
        return False, result["error"]
    return True, result.get("value")


def _exec_sandboxed(code_text, timeout_s=DEFAULT_TIMEOUT_S):
    """Executes code_text in a fresh sandboxed namespace. Returns
    (ok, namespace_or_exception)."""
    def run():
        ns = {"__builtins__": _sandbox_builtins()}
        # This corpus's snippets routinely contain top-level print() calls (it's
        # "realistic" example code) -- without suppression, every sample's exec()
        # writes straight to the real stdout, and at hundreds of samples this is
        # enough raw IOPub traffic over a remote/tunneled kernel connection (e.g.
        # VS Code -> Colab) to stall or drop the connection outright. Discarded,
        # not surfaced -- nothing here should ever depend on the executed code's
        # printed output, only its return value / raised exception.
        with _contextlib.redirect_stdout(_io.StringIO()), _contextlib.redirect_stderr(_io.StringIO()):
            exec(compile(code_text, "<smoke_test>", "exec"), ns)
        return ns
    return _run_with_timeout(run, timeout_s)


_ORDERING_SENTINEL = 1  # _DuckDummy behaves AS IF it were this number for ordering/format


class _DuckDummy:
    """Universal duck-typed placeholder for a fabricated argument. Replaces
    an earlier MagicMock-based design that turned out to be
    fundamentally the wrong tool: MagicMock's magic-method configuration
    (`m.__lt__ = ...`) only works at the INSTANCE level it's applied to --
    any CHILD mock auto-generated via attribute access, item access, or
    iteration is a fresh, unconfigured MagicMock that does NOT inherit the
    parent's configuration. Since this corpus's dominant pattern is exactly
    a NESTED comparison (`item['price'] > threshold`, `self.total > 0`),
    the previous fix (configuring ordering on the top-level dummy only)
    verified fine in isolation but still crashed on real data -- confirmed
    directly: `dummy.foo > 100` and `next(iter(dummy))['price'] > 100`
    both still raised the exact TypeError the fix was supposed to remove,
    even with the parent correctly configured. Subclassing MagicMock and
    defining `__lt__` as a normal method doesn't work either (verified) --
    MagicMock's magic-method support is genuinely special-cased and
    doesn't follow normal class-attribute-lookup semantics.

    `_DuckDummy` sidesteps this entirely by being a normal Python class:
    every dunder is a real method defined once on the class, so normal
    Python attribute/type lookup means EVERY instance (including ones
    created recursively by `__getattr__`/`__getitem__`/`__iter__`/
    arithmetic below) automatically has full behavior, with no per-instance
    configuration step to forget. Also fixes a related, previously
    undiscovered gap: MagicMock's default `__format__` only accepts an
    empty format spec, so `f"{x:.2f}"` (common in this corpus, e.g.
    `f"Order {order['id']} total: ${total:.2f}"`) raised TypeError even
    though it never showed up as ordering-comparison error text -- unlike
    ordering, `__format__` falls back to behaving as _ORDERING_SENTINEL for
    ANY format spec it can't satisfy generically, not just when the value
    is already numeric-looking, matching every other operation's "duck as a
    small number when in doubt" default."""

    def __init__(self, label="dummy"):
        self._label = label

    def __repr__(self):
        return f"<dummy:{self._label}>"

    def __str__(self):
        return f"<dummy:{self._label}>"

    def __format__(self, spec):
        if not spec:
            return str(self)
        try:
            return format(_ORDERING_SENTINEL, spec)
        except (ValueError, TypeError):
            return str(self)

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)  # let real dunder protocol lookups fail normally
        return _DuckDummy(f"{self._label}.{attr}")

    def __call__(self, *args, **kwargs):
        return _DuckDummy(f"{self._label}()")

    def __getitem__(self, key):
        return _DuckDummy(f"{self._label}[{key!r}]")

    def __setitem__(self, key, value):
        pass

    def __contains__(self, item):
        return False

    def __iter__(self):
        return iter([_DuckDummy(f"{self._label}_item0"), _DuckDummy(f"{self._label}_item1")])

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def items(self):
        return [(_DuckDummy(f"{self._label}_key0"), _DuckDummy(f"{self._label}_val0"))]

    def get(self, key, default=None):
        return _DuckDummy(f"{self._label}.get")

    def keys(self):
        return [_DuckDummy(f"{self._label}_key0")]

    def values(self):
        return [_DuckDummy(f"{self._label}_val0")]

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        return _ORDERING_SENTINEL < other if isinstance(other, (int, float)) else False

    def __gt__(self, other):
        return _ORDERING_SENTINEL > other if isinstance(other, (int, float)) else False

    def __le__(self, other):
        return _ORDERING_SENTINEL <= other if isinstance(other, (int, float)) else False

    def __ge__(self, other):
        return _ORDERING_SENTINEL >= other if isinstance(other, (int, float)) else False

    def __add__(self, other):
        return _DuckDummy(f"{self._label}+")

    __radd__ = __add__

    def __sub__(self, other):
        return _DuckDummy(f"{self._label}-")

    __rsub__ = __sub__

    def __mul__(self, other):
        return _DuckDummy(f"{self._label}*")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return _DuckDummy(f"{self._label}/")

    __rtruediv__ = __truediv__

    def __floordiv__(self, other):
        return _DuckDummy(f"{self._label}//")

    def __mod__(self, other):
        return _DuckDummy(f"{self._label}%")

    def __neg__(self):
        return _DuckDummy(f"-{self._label}")


def _make_generic_dummy(label):
    """The fallback for any parameter whose name matches nothing more
    specific below, and for self/cls -- see _DuckDummy."""
    return _DuckDummy(label)


def _make_collection_dummy(label):
    """A _DuckDummy for this corpus's 'items'/'data'/'config'-style params,
    since name alone can't tell whether the real usage is list-like
    (`for item in items: item['price']`) or dict-like (`items['key']`,
    `items.get(...)`, `items.items()`) -- _DuckDummy already tolerates
    both uniformly (and recursively, unlike the MagicMock-based version
    this replaced -- see _DuckDummy's docstring), so this is just a
    labeling convenience, not a different mechanism."""
    return _DuckDummy(label)


def _fabricate_arg(name, index):
    """Best-effort dummy value for a positional parameter, chosen by a cheap
    name-based heuristic -- falls back to a generic MagicMock, which
    tolerates almost any usage pattern (attribute access, indexing, calls)
    without crashing. Not trying to be a real input-synthesis system
    (that's a much bigger project); this only
    needs to be good enough that a call doesn't crash for an unrelated
    reason (wrong dummy shape) and masquerade as a real crash.

    KNOWN PRECISION LIMIT: `items`/`list`/`data`-style params get a
    collection-like dummy (see _make_collection_dummy) rather than a real
    list/dict, specifically so a wrong list-vs-dict guess doesn't crash.
    But MagicMock auto-implements arithmetic dunders, so `sum(items)` over
    its elements returns *another* MagicMock, not a real int -- an
    arithmetic-heavy function reached via this path won't show as cleanly
    typed as gold's real computation would. This does not affect the
    #028-motivating case: that one is a flat, self-contained script (real
    literal data, e.g. a hardcoded `pd.DataFrame({...})`) scored via
    _smoke_test_script, which never calls this function at all --
    confirmed with a stdlib-only reproduction (gold_type='int',
    pred_type='list', no MagicMock contamination). Only the dummy-args
    function-call path (_call_function_smoke/_smoke_test_function_value,
    used for extract_method/deduplicate/rename and simplify's
    single-function shape) is subject to this limit.

    The unmatched-name fallback (last line) is an ordering-tolerant
    MagicMock (_make_generic_dummy), not a plain int -- see
    _tolerate_ordering's docstring for why a plain int was tried first and
    made things worse, not better, on real data."""
    lname = name.lower()
    if lname in ("self", "cls"):
        return _make_generic_dummy(f"dummy_{lname}")
    if any(k in lname for k in ("items", "list", "rows", "records", "data", "values")):
        return _make_collection_dummy(f"dummy_{name}")
    if any(k in lname for k in ("dict", "map", "config", "options", "kwargs", "params")):
        return _make_collection_dummy(f"dummy_{name}")
    if any(k in lname for k in ("count", "num", "index", "idx", "id", "size", "len",
                                "quantity", "price", "total", "value", "amount")) or lname == "n":
        return 1
    if any(k in lname for k in ("name", "str", "text", "key", "path", "url", "status", "message")):
        return "x"
    if any(k in lname for k in ("flag", "is_", "has_", "enable")):
        return True
    return _make_generic_dummy(f"dummy_{name}_{index}")


def _call_function_smoke(module_text, func_name, timeout_s=DEFAULT_TIMEOUT_S):
    """Execs module_text (a full before/pred/gold snippet) in the sandbox,
    looks up func_name (module-level def, or a method on a module-level
    class), calls it with fabricated dummy args, and reports whether it
    crashed and what type it returned. Returns a dict: {"ok": bool,
    "result_type": str or None, "error": str or None} -- "ok" is False for
    both exec failures and call failures/timeouts, "error" carries which."""
    ok, ns_or_err = _exec_sandboxed(module_text, timeout_s)
    if not ok:
        return {"ok": False, "result_type": None, "error": f"module exec failed: {ns_or_err!r}"}
    ns = ns_or_err

    func = _get_module_func(ns, func_name)
    if func is None or not callable(func):
        return {"ok": False, "result_type": None,
                "error": f"{func_name!r} not found or not callable after exec"}

    args = _fabricate_args_for_params(_signature_params(func))
    ok2, result_or_err = _run_with_timeout(lambda: func(*args), timeout_s)
    if not ok2:
        return {"ok": False, "result_type": None, "error": repr(result_or_err)}
    return {"ok": True, "result_type": type(result_or_err).__name__, "error": None}


def _get_module_func(ns, func_name):
    """Looks up func_name in an exec'd namespace: a module-level def, or a
    method on a module-level class (getattr on the class returns the plain
    unbound function in Python 3, callable as func(dummy_self, ...))."""
    func = ns.get(func_name)
    if func is None:
        for v in ns.values():
            if isinstance(v, type) and hasattr(v, func_name):
                func = getattr(v, func_name)
                break
    return func


def _signature_params(func):
    try:
        return list(_inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return []


def _fabricate_args_for_params(params):
    args = []
    for i, p in enumerate(params):
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not p.empty:
            continue
        args.append(_fabricate_arg(p.name, i))
    return args


def _call_with_fixed_args(module_text, func_name, args, timeout_s=DEFAULT_TIMEOUT_S):
    """Like _call_function_smoke, but calls func_name with the given
    ALREADY-fabricated positional args instead of fabricating its own --
    needed when the exact same argument objects must be passed to two
    different modules' functions for a meaningful value-level comparison
    (see _smoke_test_function_value/smoke_test_rename below), unlike
    _call_function_smoke's independent-per-side fabrication, which is only
    good enough for a type-only comparison. Returns a dict including the
    raw "result" value (not just its type name), so the caller can compare
    values, not just types."""
    ok, ns_or_err = _exec_sandboxed(module_text, timeout_s)
    if not ok:
        return {"ok": False, "result": None, "result_type": None,
                "error": f"module exec failed: {ns_or_err!r}"}
    ns = ns_or_err
    func = _get_module_func(ns, func_name)
    if func is None or not callable(func):
        return {"ok": False, "result": None, "result_type": None,
                "error": f"{func_name!r} not found or not callable after exec"}
    ok2, result_or_err = _run_with_timeout(lambda: func(*args), timeout_s)
    if not ok2:
        return {"ok": False, "result": None, "result_type": None, "error": repr(result_or_err)}
    return {"ok": True, "result": result_or_err,
            "result_type": type(result_or_err).__name__, "error": None}


def _values_match(a, b):
    """True if two smoke-test call results should be considered equal.
    Falls back to type-only comparison when either value is a _DuckDummy --
    two independently-created dummies are only == by identity (`self is
    other`, see _DuckDummy.__eq__), so two "equivalent" dummies from
    separate calls are never really equal even when they'd represent the
    same conceptual value -- only a non-dummy value gets compared for real
    equality.

    Recurses into list/tuple/dict rather than delegating straight to `==`
    for these -- found the hard way (sample #006 in a live
    Colab run): a genuinely correct rename whose function returns a LIST
    OF _DuckDummy elements (e.g. `[x.strip() for x in items]`) was flagged
    as a false-positive `value_mismatch`, because the outer value is a real
    `list` (so the top-level _DuckDummy check above doesn't trigger), and
    plain `list.__eq__` compares elements pairwise via `_DuckDummy.__eq__`
    -- identity-based, so two independently-fabricated dummies representing
    the "same" conceptual element are never equal, even for byte-identical
    code. Recursing through `_values_match` itself for list/tuple/dict
    elements means a _DuckDummy nested at any depth gets the same
    type-only tolerance the top-level check already gave it. Sets are
    deliberately NOT recursed into (same identity-comparison problem would
    apply, and set membership under a custom `__hash__`/`__eq__` is murkier
    to reason about) -- falls through to length-only comparison for those.

    Wrapped in try/except since an arbitrary returned object's own __eq__
    could itself raise (e.g. a pandas Series/DataFrame's __eq__ returns an
    element-wise object, not a bool, which bool() on it can raise for)."""
    if isinstance(a, _DuckDummy) or isinstance(b, _DuckDummy):
        return type(a) is type(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(_values_match(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_match(a[k], b[k]) for k in a)
    if isinstance(a, set) and isinstance(b, set):
        return len(a) == len(b)
    try:
        return bool(a == b)
    except Exception:
        return type(a) is type(b)


def _smoke_test_function(gold_text, gold_func_name, pred_text, pred_func_name,
                         timeout_s=DEFAULT_TIMEOUT_S):
    """Calls gold's and pred's respective (possibly differently-named --
    name-agnostic, matching every other scorer's design) function and
    compares the two. smoke_ok is None (inconclusive, not a pred failure)
    if gold itself couldn't establish a baseline -- e.g. it genuinely needs
    an import this sandbox blocked, or doesn't run standalone."""
    gold_res = _call_function_smoke(gold_text, gold_func_name, timeout_s)
    pred_res = _call_function_smoke(pred_text, pred_func_name, timeout_s)
    if not gold_res["ok"]:
        return {"smoke_ok": None, "gold": gold_res, "pred": pred_res, "reason": "gold_baseline_failed"}
    if not pred_res["ok"]:
        return {"smoke_ok": False, "gold": gold_res, "pred": pred_res, "reason": "pred_crashed"}
    match = gold_res["result_type"] == pred_res["result_type"]
    return {"smoke_ok": match, "gold": gold_res, "pred": pred_res,
            "reason": None if match else "return_type_mismatch"}


def _smoke_test_function_value(before_text, before_func_name, pred_text, pred_func_name,
                               timeout_s=DEFAULT_TIMEOUT_S):
    """Like _smoke_test_function, but for rename's correctness criterion: a
    well-formed rename is a pure syntactic transformation, so before's and
    pred's function should compute the SAME VALUE, not just the same type
    -- a strictly stronger check than every other smoke test in this
    module, which must tolerate a genuinely different valid answer.

    Fabricates dummy args ONCE from before's own signature and calls BOTH
    before's and pred's function with the exact same argument objects
    (matched positionally, since the two functions' parameter names may
    differ after the rename) -- unlike _call_function_smoke's independent-
    per-side fabrication, which can only support a type-only comparison."""
    ok, ns_or_err = _exec_sandboxed(before_text, timeout_s)
    if not ok:
        return {"smoke_ok": None, "reason": "before_baseline_failed", "detail": repr(ns_or_err)}
    before_func = _get_module_func(ns_or_err, before_func_name)
    if before_func is None or not callable(before_func):
        return {"smoke_ok": None, "reason": "before_func_not_found"}

    args = _fabricate_args_for_params(_signature_params(before_func))

    ok_b, before_result = _run_with_timeout(lambda: before_func(*args), timeout_s)
    if not ok_b:
        return {"smoke_ok": None, "reason": "before_baseline_failed", "detail": repr(before_result)}

    pred_res = _call_with_fixed_args(pred_text, pred_func_name, args, timeout_s)
    if not pred_res["ok"]:
        return {"smoke_ok": False, "reason": "pred_crashed",
                "before_type": type(before_result).__name__, "pred": pred_res}

    match = _values_match(before_result, pred_res["result"])
    return {"smoke_ok": match, "reason": None if match else "value_mismatch",
            "before_type": type(before_result).__name__, "pred_type": pred_res["result_type"]}


def _smoke_test_script_value(before_text, pred_text, timeout_s=DEFAULT_TIMEOUT_S):
    """Flat-script counterpart to _smoke_test_function_value: execs
    before_text and pred_text, compares the runtime VALUE (not just type)
    of the last top-level-assigned variable in each -- looked up
    separately by name on each side (rename may have renamed the target
    variable itself, same positional-not-name-based matching philosophy as
    the function-call path above). Meaningful for a flat self-contained
    script (literal data, no dummy args needed): before and pred should
    produce identical output regardless of which identifier names got it
    there."""
    try:
        before_tree = _ast.parse(before_text)
    except SyntaxError:
        return {"smoke_ok": None, "reason": "before_parse_fail"}
    before_target = _last_top_level_assigned_name(before_tree)
    if before_target is None:
        return {"smoke_ok": None, "reason": "no_target_variable"}

    try:
        pred_tree = _ast.parse(pred_text)
    except SyntaxError:
        return {"smoke_ok": False, "reason": "pred_parse_fail"}
    pred_target = _last_top_level_assigned_name(pred_tree) or before_target

    ok_b, ns_b = _exec_sandboxed(before_text, timeout_s)
    if not ok_b:
        return {"smoke_ok": None, "reason": "before_baseline_failed", "detail": repr(ns_b)}
    if before_target not in ns_b:
        return {"smoke_ok": None, "reason": "before_baseline_failed",
                "detail": f"target variable {before_target!r} not found in before's own namespace after exec"}

    ok_p, ns_p = _exec_sandboxed(pred_text, timeout_s)
    if not ok_p:
        return {"smoke_ok": False, "reason": "pred_crashed", "error": repr(ns_p)}
    if pred_target not in ns_p:
        return {"smoke_ok": False, "reason": "pred_missing_target_variable"}

    before_value = ns_b[before_target]
    pred_value = ns_p[pred_target]
    match = _values_match(before_value, pred_value)
    return {"smoke_ok": match, "reason": None if match else "value_mismatch",
            "before_target": before_target, "pred_target": pred_target,
            "before_type": type(before_value).__name__, "pred_type": type(pred_value).__name__}


def smoke_test_rename(before_text, pred_text, gold_text, timeout_s=DEFAULT_TIMEOUT_S):
    """Opt-in execution-based smoke test for rename. Compares pred against
    BEFORE (not gold, unlike every other smoke test here) -- gold's chosen
    new names are irrelevant to whether the rename itself is correct, and a
    well-formed rename should preserve before's behavior exactly, value for
    value, regardless of which names either side picked. `gold_text` is
    accepted (unused) only so this function has the same call signature as
    every other entry in SMOKE_TEST_SCORERS.

    Only handles the single-function shape (_func_names(before_text) and
    _func_names(pred_text) both return exactly one name) -- same scoping-
    down as extract_method's K==1 restriction and for the same reason: with
    multiple functions/methods in scope there's no reliable way to know
    which one is "the" renamed target without duplicating score_rename's
    positional NAME-token matching, not worth it for a cheap smoke-test
    tier. Falls back to a flat-script comparison (_smoke_test_script_value)
    when before_text has no top-level function at all. Returns
    smoke_ok=None (inconclusive) for every other shape (e.g. a class with
    multiple methods)."""
    if pred_text is None:
        return {"smoke_ok": None, "reason": "pred_parse_fail"}

    before_funcs = _func_names(before_text)
    pred_funcs = _func_names(pred_text)
    if before_funcs is None or pred_funcs is None:
        return {"smoke_ok": None, "reason": "parse_fail"}

    if len(before_funcs) == 1 and len(pred_funcs) == 1:
        return _smoke_test_function_value(before_text, next(iter(before_funcs)),
                                          pred_text, next(iter(pred_funcs)), timeout_s)
    if len(before_funcs) == 0 and len(pred_funcs) == 0:
        return _smoke_test_script_value(before_text, pred_text, timeout_s)
    return {"smoke_ok": None, "reason": "not_single_function_shape"}


def _last_top_level_assigned_name(tree):
    """Name of the last variable assigned at module top level in tree, or
    None. A cheap proxy for "the value the script actually produces" in a
    flat (non-function) snippet -- typically what a trailing print()/return
    would reference."""
    name = None
    for node in tree.body:
        if isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    name = t.id
        elif isinstance(node, _ast.AnnAssign) and isinstance(node.target, _ast.Name):
            name = node.target.id
    return name


def _smoke_test_script(gold_text, pred_text, timeout_s=DEFAULT_TIMEOUT_S):
    """For a flat (non-function) before/after script: execs both pred and
    gold in the sandbox and compares the runtime type of the last
    top-level-assigned variable -- catches e.g. #028 (`active_count_sum`):
    gold ends up a scalar, pred ends up a DataFrame, a type mismatch
    invisible to every AST/text-based scorer in scorers.py."""
    try:
        gold_tree = _ast.parse(gold_text)
    except SyntaxError:
        return {"smoke_ok": None, "reason": "gold_parse_fail"}
    target = _last_top_level_assigned_name(gold_tree)
    if target is None:
        return {"smoke_ok": None, "reason": "no_target_variable"}

    ok_g, ns_or_err_g = _exec_sandboxed(gold_text, timeout_s)
    if not ok_g:
        return {"smoke_ok": None, "reason": "gold_baseline_failed", "detail": repr(ns_or_err_g)}
    if target not in ns_or_err_g:
        return {"smoke_ok": None, "reason": "gold_baseline_failed",
                "detail": f"target variable {target!r} not found in gold's own namespace after exec"}

    ok_p, ns_or_err_p = _exec_sandboxed(pred_text, timeout_s)
    if not ok_p:
        return {"smoke_ok": False, "reason": "pred_crashed", "error": repr(ns_or_err_p)}
    if target not in ns_or_err_p:
        return {"smoke_ok": False, "reason": "pred_missing_target_variable"}

    gold_type = type(ns_or_err_g[target]).__name__
    pred_type = type(ns_or_err_p[target]).__name__
    match = gold_type == pred_type
    return {"smoke_ok": match, "gold_type": gold_type, "pred_type": pred_type,
            "target": target, "reason": None if match else "return_type_mismatch"}


def smoke_test_simplify(before_text, pred_text, gold_text, timeout_s=DEFAULT_TIMEOUT_S):
    """Opt-in execution-based smoke test for simplify.
    If gold's AFTER text defines exactly one top-level function that pred
    also defines (by the same name -- simplify doesn't rename), calls both
    with fabricated dummy args and compares return types. Otherwise treats
    it as a flat script and compares the runtime type of the last
    top-level-assigned variable (_smoke_test_script)."""
    if pred_text is None:
        return {"smoke_ok": None, "reason": "pred_parse_fail"}
    try:
        gold_tree = _ast.parse(gold_text)
    except SyntaxError:
        return {"smoke_ok": None, "reason": "gold_parse_fail"}
    top_funcs = [n.name for n in gold_tree.body
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))]
    if len(top_funcs) == 1:
        func_name = top_funcs[0]
        try:
            pred_tree = _ast.parse(pred_text)
        except SyntaxError:
            return {"smoke_ok": False, "reason": "pred_parse_fail"}
        pred_top_funcs = [n.name for n in pred_tree.body
                         if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))]
        if func_name in pred_top_funcs:
            return _smoke_test_function(gold_text, func_name, pred_text, func_name, timeout_s)
    return _smoke_test_script(gold_text, pred_text, timeout_s)


def smoke_test_extract_method_or_dedup(before_text, pred_text, gold_text, timeout_s=DEFAULT_TIMEOUT_S):
    """Opt-in execution-based smoke test for the single-new-function shape of
    extract_method/deduplicate: calls gold's and
    pred's newly-introduced function (name-agnostic -- looked up separately
    by each side's own name) with fabricated dummy args and compares (a)
    neither crashes and (b) the return types match.

    Only handles the K==1 shape (gold introduces exactly one new function)
    -- the K>=2 shape would need this module to duplicate scorers.py's
    body-similarity matching to know which pred helper corresponds to which
    gold helper, not worth it for a cheap smoke-test tier. Returns
    smoke_ok=None (inconclusive) for every other shape."""
    if pred_text is None:
        return {"smoke_ok": None, "reason": "pred_parse_fail"}
    before_funcs = _func_names(before_text)
    pred_funcs = _func_names(pred_text)
    gold_funcs = _func_names(gold_text)
    if any(x is None for x in (before_funcs, pred_funcs, gold_funcs)):
        return {"smoke_ok": None, "reason": "parse_fail"}
    gold_new = gold_funcs - before_funcs
    pred_new = pred_funcs - before_funcs
    if len(gold_new) != 1 or len(pred_new) != 1:
        return {"smoke_ok": None, "reason": "not_single_helper_shape"}
    return _smoke_test_function(gold_text, next(iter(gold_new)),
                                pred_text, next(iter(pred_new)), timeout_s)


# Deliberately separate from scorers.py's TASK_SCORERS -- opt-in only, same
# discipline as TASK_SCORERS_SEMANTIC. remove_dead_code is excluded: dead
# code by definition never executes, so a runtime type/value comparison is
# blind to the dominant failure mode (leftover dead code changes nothing
# about runtime behavior) and the other failure mode (removing live code)
# is already caught more precisely by score_remove_dead_code's exact
# line-index diff (already audited clean) -- there's nothing this tier can
# add there. rename IS included (smoke_test_
# rename) -- unlike every other entry here, it compares pred against
# BEFORE, by VALUE not just type, since a well-formed rename should
# preserve behavior exactly.
SMOKE_TEST_SCORERS = {
    "rename": smoke_test_rename,
    "simplify": smoke_test_simplify,
    "extract_method": smoke_test_extract_method_or_dedup,
    "deduplicate": smoke_test_extract_method_or_dedup,
}


def smoke_test_pool_worker(args):
    """multiprocessing.Pool worker target for running SMOKE_TEST_SCORERS at
    scale (e.g. a full 750-sample held-out split, not just a ~150-sample
    validation pass). Must be a module-level function (not a notebook-local
    closure) so it can be pickled and dispatched to a forked/spawned child
    process.

    Why this exists: _run_with_timeout's own docstring already documents that
    a timed-out exec's thread is abandoned, never actually killed (Python has
    no safe way to force-kill a thread) -- an accepted limitation for a
    best-effort tier used in small batches, but NOT safe in a tight loop over
    hundreds of untrusted samples, since every hang leaves one more thread
    permanently spinning in the caller's process. Confirmed empirically:
    a real 750-sample run showed rename and simplify complete
    cleanly (no per-task bug), then deduplicate died ~175 samples in -- not a
    task-boundary bug, but cumulative thread accumulation across the whole
    session finally exhausting the process, with whichever task happened to
    be running at that point looking like "the cause."

    Run each sample through this function via
    multiprocessing.Pool(processes=1, maxtasksperchild=N) instead of calling
    SMOKE_TEST_SCORERS directly in-process: the OS kills and respawns the
    worker process every N tasks, which reliably reaps every thread that
    worker ever spawned -- hung or not -- bounding the maximum number of
    ever-accumulated stuck threads to at most one recycle interval's worth,
    instead of letting them pile up for an entire run's lifetime.

    args: (ttype, before_text, pred_text, gold_text, timeout_s) -- a plain
    tuple of primitives/strings, safe to pickle across the process boundary
    (unlike the sandboxed exec's own internal namespace, which can contain
    unpicklable objects -- this only ever crosses the boundary as the already-
    finished result dict, never as live sandboxed state)."""
    ttype, before_text, pred_text, gold_text, timeout_s = args
    scorer = SMOKE_TEST_SCORERS.get(ttype)
    if scorer is None:
        return {"smoke_ok": None, "reason": "no_smoke_scorer_for_type"}
    try:
        return scorer(before_text, pred_text, gold_text, timeout_s)
    except Exception as e:  # noqa: BLE001 -- worker-boundary safety net; the scorer's own
        # internal sandboxing already catches everything from the executed code itself,
        # this only guards against something unexpected in the scorer's own control flow
        return {"smoke_ok": None, "reason": "worker_exception", "detail": repr(e)}
