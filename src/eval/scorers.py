## non_interference = 1.0 - touched / max(n, 1)


import ast as _ast
import builtins as _builtins
import copy as _copy
import tokenize as _tokenize
import io as _io
from difflib import SequenceMatcher


# -- rename: name-agnostic, continuous --------------------------------------
def _normalize_code(text):
    try:
        return _ast.unparse(_ast.parse(text))
    except SyntaxError:
        return None


def _name_tokens(text):
    toks = []
    try:
        for tok in _tokenize.generate_tokens(_io.StringIO(text).readline):
            if tok.type == _tokenize.NAME:
                toks.append(tok.string)
    except _tokenize.TokenError:
        return None
    return toks


def _gold_rename_complexity(before_text, gold_text):
    """n_sites (# positions gold renamed) from before/gold alone -- independent
    of whether pred parses. Returns None only when before/gold themselves fail
    to normalize or their name-token counts don't align (a gold-data issue,
    not a pred failure)."""
    gold_norm = _normalize_code(gold_text)
    before_norm = _normalize_code(before_text)
    if gold_norm is None or before_norm is None:
        return None
    before_names = _name_tokens(before_norm)
    gold_names = _name_tokens(gold_norm)
    if before_names is None or gold_names is None:
        return None
    if len(before_names) != len(gold_names):
        return None
    return sum(1 for a, b in zip(before_names, gold_names) if a != b)


def find_pred_collisions(before_text, pred_text):
    """Detect the injectivity violation directly from before/pred, no gold
    needed (usable at real inference time, not just eval) -- two different
    original identifiers in `before` that pred renamed to the identical new
    name. Returns a set of the colliding new-name strings (each claimed by 2+
    distinct old names), or an empty set if none/unparseable.
    """
    before_norm = _normalize_code(before_text)
    pred_norm = _normalize_code(pred_text)
    if before_norm is None or pred_norm is None:
        return set()
    before_names = _name_tokens(before_norm)
    pred_names = _name_tokens(pred_norm)
    if before_names is None or pred_names is None:
        return set()

    old_to_news = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, before_names, pred_names, autojunk=False).get_opcodes():
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue
        for oi, pi in zip(range(i1, i2), range(j1, j2)):
            old_to_news.setdefault(before_names[oi], set()).add(pred_names[pi])

    new_to_olds = {}
    for old_name, news in old_to_news.items():
        for new_name in news:
            new_to_olds.setdefault(new_name, set()).add(old_name)

    return {new_name for new_name, olds in new_to_olds.items() if len(olds) > 1}


def score_rename(before_text, pred_text, gold_text):
    """Name-agnostic, continuous rename score. The model's chosen new name need
    not match gold's -- only needs to be applied consistently at every site gold
    renamed, without touching anything else (behavior preservation).

    Returns dict: score, localization_recall, consistency, non_interference,
    executable, complexity (n_sites = # occurrences gold renamed), reason
    (on hard fail).
    """
    n_sites_gold = _gold_rename_complexity(before_text, gold_text)

    if pred_text is None:
        return {"score": 0.0, "localization_recall": 0.0, "consistency": 0.0,
                "non_interference": 0.0, "executable": False, "duplicate_defs": False,
                "complexity": n_sites_gold, "reason": "parse_fail"}

    pred_norm = _normalize_code(pred_text)
    gold_norm = _normalize_code(gold_text)
    before_norm = _normalize_code(before_text)
    if pred_norm is None or gold_norm is None or before_norm is None:
        return {"score": 0.0, "localization_recall": 0.0, "consistency": 0.0,
                "non_interference": 0.0, "executable": False, "duplicate_defs": False,
                "complexity": n_sites_gold, "reason": "parse_fail"}

    before_names = _name_tokens(before_norm)
    pred_names = _name_tokens(pred_norm)
    gold_names = _name_tokens(gold_norm)
    if before_names is None or pred_names is None or gold_names is None:
        return {"score": 0.0, "localization_recall": 0.0, "consistency": 0.0,
                "non_interference": 0.0, "executable": False, "duplicate_defs": False,
                "complexity": n_sites_gold, "reason": "parse_fail"}

    executable = not _has_undefined_names(pred_text)
    duplicate_defs = bool(_duplicate_def_names(pred_text))

    if len(pred_names) == len(before_names) and len(gold_names) != len(before_names):
        # gold itself isn't a pure rename here -- e.g. it generalized a hardcoded
        # literal into a new parameter (`5` -> `threshold`) alongside the rename.
        similarity = SequenceMatcher(None, pred_norm, gold_norm).ratio()
        if not executable or duplicate_defs:
            similarity = min(similarity, 0.15)
        return {"score": similarity, "localization_recall": None, "consistency": None,
                "non_interference": None, "executable": executable,
                "duplicate_defs": duplicate_defs,
                "complexity": n_sites_gold, "reason": "gold_not_pure_rename"}

    if not (len(before_names) == len(pred_names) == len(gold_names)):
        return {"score": 0.0, "localization_recall": 0.0, "consistency": 0.0,
                "non_interference": 0.0, "executable": executable,
                "duplicate_defs": duplicate_defs,
                "complexity": n_sites_gold, "reason": "structural_mismatch"}

    n = len(before_names)
    gold_sites = [i for i in range(n) if gold_names[i] != before_names[i]]
    pred_sites = set(i for i in range(n) if pred_names[i] != before_names[i])
    n_sites = len(gold_sites)

    if n_sites == 0:
        touched = len(pred_sites)
        non_interference = 1.0 - touched / max(n, 1)
        score = non_interference if (executable and not duplicate_defs) else min(non_interference, 0.15)
        return {"score": score, "localization_recall": 1.0,
                "consistency": 1.0, "non_interference": non_interference,
                "executable": executable, "duplicate_defs": duplicate_defs,
                "complexity": 0, "reason": None}

    hit_sites = [i for i in gold_sites if i in pred_sites]
    localization_recall = len(hit_sites) / n_sites

    groups = {}
    for i in gold_sites:
        groups.setdefault(before_names[i], []).append(i)

    # Per-group majority vote: which new name did the model settle on for each
    # original identifier, and how many of that group's hits agree on it.
    group_choice = {}
    for old_name, idxs in groups.items():
        hits = [i for i in idxs if i in pred_sites]
        if not hits:
            continue
        chosen = {}
        for i in hits:
            chosen[pred_names[i]] = chosen.get(pred_names[i], 0) + 1
        best_name = max(chosen, key=chosen.get)
        group_choice[old_name] = (best_name, chosen[best_name])

    # Injectivity check: two *different* original identifiers collapsing onto
    # the same new name is a real semantic break (e.g. a loop variable and a
    # parameter both renamed to `item`, breaking an `item > item` comparison)
    # even though each group looks internally consistent in isolation -- don't
    # credit colliding groups as consistent renames.
    name_to_groups = {}
    for old_name, (new_name, _count) in group_choice.items():
        name_to_groups.setdefault(new_name, []).append(old_name)

    consistent_hit_count = 0
    for old_name, (new_name, count) in group_choice.items():
        if len(name_to_groups[new_name]) > 1:
            continue
        consistent_hit_count += count
    consistency = consistent_hit_count / n_sites

    non_gold_positions = [i for i in range(n) if i not in set(gold_sites)]
    if non_gold_positions:
        interfered = sum(1 for i in non_gold_positions if i in pred_sites)
        non_interference = 1.0 - interfered / len(non_gold_positions)
    else:
        non_interference = 1.0

    score = localization_recall * consistency * non_interference
    if not executable or duplicate_defs:
        # Soft cap, not a hard zero -- same reasoning/value as every other
        # scorer's _has_undefined_names gate: it's a snippet-local heuristic
        # that can false-positive on legitimate outer-scope references, so a
        # hard 0/1 gate here would reintroduce discontinuity this scorer was
        # built to avoid. Catches an inconsistent rename that leaves a real
        # NameError behind, and duplicate_defs (two renamed methods landing on
        # the same new name -- the second silently shadows the first).
        score = min(score, 0.15)
    return {"score": score, "localization_recall": localization_recall,
            "consistency": consistency, "non_interference": non_interference,
            "executable": executable, "duplicate_defs": duplicate_defs,
            "complexity": n_sites, "reason": None}


# -- extract_method: snippet + call-site, name-agnostic ----------------------
def _func_names(code_text):
    """Return set of all function names in code_text, or None on SyntaxError."""
    try:
        tree = _ast.parse(code_text)
        return {n.name for n in _ast.walk(tree)
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    except SyntaxError:
        return None


def _rename_func(code_text, old_name, new_name):
    """Rename every reference to old_name -> new_name via AST. None on failure."""
    try:
        tree = _ast.parse(code_text)
    except SyntaxError:
        return None

    class _R(_ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == old_name:
                node.name = new_name
            self.generic_visit(node)
            return node
        def visit_AsyncFunctionDef(self, node):
            if node.name == old_name:
                node.name = new_name
            self.generic_visit(node)
            return node
        def visit_Name(self, node):
            if node.id == old_name:
                node.id = new_name
            return node

    try:
        renamed = _R().visit(tree)
        _ast.fix_missing_locations(renamed)
        return _ast.unparse(renamed)
    except Exception:
        return None


CANONICAL = "_extracted_"

# See _score_extract_method_multi's pred_new cap: real gold data (LLM-generated,
# ~10-20 line snippets) never introduces more than a handful of new functions --
# anything past this is degenerate decode output, not a plausible extraction.
MAX_PRED_NEW_FUNCS = 20


def _find_named_func(tree, name):
    """Locates a FunctionDef/AsyncFunctionDef by name anywhere in the tree
    (ast.walk), not just at module top-level -- the extracted function may be
    a class method (OOP-layout records), not a top-level def."""
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _func_scope_kind(code_text, func_name):
    """Enclosing-scope kind ('Module', 'ClassDef', or 'FunctionDef') of the
    first function named func_name found anywhere in code_text, or None if
    not found or code_text fails to parse.

    _find_named_func's ast.walk (used throughout this module to locate the
    extracted/matched helper) deliberately finds a same-named def anywhere in
    the tree, to support the legitimate case where the extracted helper is a
    class method. But that means it never checks *where* it found it: a `def`
    that should sit at module/class level but ends up nested one level too
    deep inside another function's body (a bad SEARCH/REPLACE hunk boundary)
    still parses as valid Python while the nested function is actually
    unreachable/broken. This check compares gold's and pred's scope kind for
    the same helper name and flags a mismatch."""
    try:
        tree = _ast.parse(code_text)
    except SyntaxError:
        return None
    node = _find_named_func(tree, func_name)
    if node is None:
        return None
    parent = _build_parent_map(tree).get(node)
    if isinstance(parent, _ast.ClassDef):
        return "ClassDef"
    if isinstance(parent, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        return "FunctionDef"
    return "Module"


def _existing_func_became_nested(before_text, pred_text):
    """True if any function present in BOTH before_text and pred_text (i.e.
    one pred didn't rename or newly extract -- an existing function that
    should just carry over as-is) ends up nested inside another function's
    body in pred when it wasn't nested in before.

    This is the *other* shape of a bad-hunk-boundary corruption, not caught
    by _func_scope_kind's gold-relative comparison above: an EXISTING function
    untouched by the extraction ends up nested INSIDE the new helper instead
    of staying a sibling. None of this project's five task types ever
    legitimately nests a previously free-standing function inside another
    function's body, so this needs no gold reference -- it's a pure
    before-vs-pred structural-integrity check, the same spirit as
    find_pred_collisions for rename."""
    try:
        before_tree = _ast.parse(before_text)
        pred_tree = _ast.parse(pred_text)
    except SyntaxError:
        return False
    before_funcs = {n.name for n in _ast.walk(before_tree)
                    if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    pred_funcs = {n.name for n in _ast.walk(pred_tree)
                 if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    shared = before_funcs & pred_funcs
    if not shared:
        return False

    before_parents = _build_parent_map(before_tree)
    pred_parents = _build_parent_map(pred_tree)
    for name in shared:
        b_node = _find_named_func(before_tree, name)
        p_node = _find_named_func(pred_tree, name)
        if b_node is None or p_node is None:
            continue
        b_nested = isinstance(before_parents.get(b_node), (_ast.FunctionDef, _ast.AsyncFunctionDef))
        p_nested = isinstance(pred_parents.get(p_node), (_ast.FunctionDef, _ast.AsyncFunctionDef))
        if p_nested and not b_nested:
            return True
    return False


class _RemoveNamedFunc(_ast.NodeTransformer):
    """Removes every FunctionDef/AsyncFunctionDef with the given name,
    wherever it lives (module top-level or nested inside a class body)."""
    def __init__(self, name):
        self.name = name
    def visit_FunctionDef(self, node):
        if node.name == self.name:
            return None
        self.generic_visit(node)
        return node
    def visit_AsyncFunctionDef(self, node):
        if node.name == self.name:
            return None
        self.generic_visit(node)
        return node


def _split_extracted(tree):
    """Return (extracted_funcdef_node, [other_top_level_nodes]) for a module
    whose CANONICAL-named function is the extracted helper. `others` is built
    from an independent deep copy so removing the extracted node doesn't
    mutate the tree `extracted` was taken from. `extracted` can be None if the
    CANONICAL function genuinely isn't found -- callers must check for this."""
    extracted = _find_named_func(tree, CANONICAL)
    others_tree = _RemoveNamedFunc(CANONICAL).visit(_copy.deepcopy(tree))
    _ast.fix_missing_locations(others_tree)
    return extracted, others_tree.body


def _extracted_helper_text(full_text, name):
    """Name-agnostic source text of the function `name` in full_text: rename it
    (and all references) to CANONICAL, then unparse just that FunctionDef.
    Returns None on any parse/rename failure. This is the per-helper primitive
    the single- and multi-extraction paths share."""
    renamed = _rename_func(full_text, name, CANONICAL)
    if renamed is None:
        return None
    try:
        tree = _ast.parse(renamed)
    except SyntaxError:
        return None
    node = _find_named_func(tree, CANONICAL)
    if node is None:
        return None
    return _ast.unparse(node)


def _strip_named_funcs(full_text, names):
    """Everything in full_text except the top-level/nested defs whose names are
    in `names` (the extracted helpers), unparsed back to text -- the "remainder"
    (caller + call sites) used for the callsite comparison. None on parse fail."""
    try:
        tree = _ast.parse(full_text)
    except SyntaxError:
        return None
    for name in names:
        tree = _RemoveNamedFunc(name).visit(tree)
    _ast.fix_missing_locations(tree)
    return "\n".join(_ast.unparse(n) for n in tree.body)


def _gold_extract_method_n_lines(before_text, gold_text):
    """Total extracted-helper-body line count from before/gold alone --
    independent of whether pred parses (see score_rename's
    _gold_rename_complexity for the same fix and why it matters: a total
    apply_sr failure previously discarded complexity info that was already fully
    determined by before/gold). Generalized to sum over all new
    functions gold introduced relative to before, so the multi-extraction shape
    reports a real complexity too. Returns None only when gold introduces zero
    new functions (nothing extracted -- genuinely undefined) or anything fails
    to parse."""
    gold_text = gold_text.strip()
    before_funcs = _func_names(before_text)
    gold_funcs = _func_names(gold_text)
    if before_funcs is None or gold_funcs is None:
        return None
    gold_new = gold_funcs - before_funcs
    if len(gold_new) == 0:
        return None
    total = 0
    for name in gold_new:
        body = _extracted_helper_text(gold_text, name)
        if body is None:
            return None
        total += len(body.splitlines())
    return total


def _score_extract_method_multi(before_text, pred_text, gold_text,
                                before_funcs, pred_funcs, gold_funcs):
    """Scores the extract_method shapes where gold introduces != 1 new function.

    K>=2 (multiple helpers extracted at once): structural, name-agnostic.
      snippet_score  -- each gold helper is best-matched (highest body-similarity)
                        to a pred helper; averaged over max(#gold, #pred) helpers
                        so both missing and spurious helpers reduce the score
                        continuously (no hard exact-count gate like the single
                        path's wrong_new_func_count).
      callsite_score -- similarity of the remainder (caller + call sites, all
                        extracted helpers stripped) of pred vs gold.
    K==0 (gold introduces no named helper -- genuinely un-scoreable as an
      extraction): falls back to whole-file similarity, flagged
      reason="no_gold_extraction" so it can be reported as its own population
      instead of the old tautological exact-string 1.0/0.0. (0 occurrences in
      the current eval set, but handled non-tautologically for completeness.)

    Also gates the final score on `executable` (pred_text has no snippet-local
    undefined-name reference, via `_has_undefined_names` -- reused as-is, see
    its own docstring) -- neither this nor the single-extraction path in
    score_extract_method previously checked this at all: a pred that extracts
    a plausible-looking helper but forgets to pass a variable it needs (a real
    NameError at the new call site) scored identically to a correct
    extraction, since snippet_score/callsite_score are pure text similarity.
    Same fix, and by construction the same one call site, benefits
    score_deduplicate (below), which reuses this function directly.

    KNOWN LIMITATION (not something a formula tweak fixes):
    snippet_score/callsite_score are character-level
    SequenceMatcher ratios, blind to small-but-semantically-critical edits.
    A helper with one flipped comparison (`i > 0` -> `i < 0`, computing a
    completely different function) scored snippet_score=0.992, callsite_score
    =1.0, overall 0.996 -- `_has_undefined_names` catches NameErrors, nothing
    syntactically valid and semantically wrong. callsite_score is also nearly
    free once the call shape is right, since the caller side doesn't encode
    anything about the callee's internals. Closing this properly needs
    execution-based scoring (run pred's/gold's code against sample inputs,
    compare outputs) -- read this scorer's mean_score as "structurally
    plausible extraction," not "verified-correct extraction."
    """
    gold_new = gold_funcs - before_funcs
    complexity = _gold_extract_method_n_lines(before_text, gold_text)
    executable = not _has_undefined_names(pred_text)
    duplicate_defs = bool(_duplicate_def_names(pred_text))

    if len(gold_new) == 0:
        s = SequenceMatcher(None, pred_text, gold_text).ratio()
        if not executable or duplicate_defs:
            # Soft cap, not a hard zero -- _has_undefined_names is a snippet-
            # local heuristic that false-positives on legitimate outer-scope
            # references (e.g. a class/helper defined elsewhere the snippet
            # calls but doesn't bind), confirmed empirically (~15-17% of
            # verbatim-gold predictions tripped a hard-zero version of this
            # gate). Same soft-cap value/reasoning as score_text_similarity.
            s = min(s, 0.15)
        return {"score": s, "snippet_score": s, "callsite_score": s,
                "complexity": complexity, "executable": executable,
                "duplicate_defs": duplicate_defs, "reason": "no_gold_extraction"}

    gold_new_list = list(gold_new)
    gold_helpers = [_extracted_helper_text(gold_text, n) for n in gold_new_list]
    if any(h is None for h in gold_helpers):
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": complexity, "executable": executable,
                "duplicate_defs": duplicate_defs, "reason": "extract_split_fail"}

    pred_new = pred_funcs - before_funcs
    if len(pred_new) > MAX_PRED_NEW_FUNCS:
        # Defensive cap against a degenerate/repeating decode (a real, documented
        # failure mode for this model on deduplicate/extract_method) hallucinating
        # dozens-to-hundreds of spurious `def` blocks: each would otherwise go
        # through a full parse+rename+reparse+unparse in _extracted_helper_text
        # below, then an O(|gold_helpers| x |pred_helpers|) SequenceMatcher
        # matching pass -- measured at 20s for a single sample at 300 spurious
        # functions. Output this degenerate deserves a hard 0 anyway (it is not
        # a plausible extraction/dedup by any reading), so skip straight there
        # instead of paying that cost to arrive at ~0 regardless.
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": complexity, "executable": executable,
                "duplicate_defs": duplicate_defs, "reason": "too_many_new_funcs"}
    pred_new_list = list(pred_new)
    pred_pairs = [(n, h) for n, h in
                  ((n, _extracted_helper_text(pred_text, n)) for n in pred_new_list)
                  if h is not None]

    nesting_mismatch = False
    if not pred_pairs:
        snippet_score = 0.0
    else:
        remaining = list(pred_pairs)
        matched = []
        for gname, gh in zip(gold_new_list, gold_helpers):
            best_i, best_r = -1, -1.0
            for i, (_pname, ph) in enumerate(remaining):
                r = SequenceMatcher(None, ph, gh).ratio()
                if r > best_r:
                    best_r, best_i = r, i
            matched.append(best_r if best_i >= 0 else 0.0)
            if best_i >= 0:
                pname, _ph = remaining.pop(best_i)
                # Nesting-shape gate , same idea as the
                # single-helper path: only flag when the matched helper is
                # genuinely nested inside another FUNCTION on one side and
                # not the other -- unreachable/broken regardless of callsite
                # adaptation. Module vs ClassDef is deliberately not flagged
                # (see score_extract_method's docstring note for the
                # adversarial case that motivated narrowing this).
                g_kind = _func_scope_kind(gold_text, gname)
                p_kind = _func_scope_kind(pred_text, pname)
                if "FunctionDef" in (g_kind, p_kind) and g_kind != p_kind:
                    nesting_mismatch = True
        denom = max(len(gold_helpers), len(pred_pairs))
        snippet_score = sum(matched) / denom

    if not nesting_mismatch and _existing_func_became_nested(before_text, pred_text):
        # Same before-vs-pred structural-integrity check as the single-helper
        # path -- catches an EXISTING function (not part of gold_new/pred_new
        # at all) getting swallowed into one of the new helpers, which the
        # matched-pair check above can't see since it only looks at the new
        # functions themselves.
        nesting_mismatch = True

    gold_others = _strip_named_funcs(gold_text, gold_new)
    pred_others = _strip_named_funcs(pred_text, pred_new)
    if gold_others is None or pred_others is None:
        callsite_score = 0.0
    else:
        callsite_score = SequenceMatcher(None, pred_others, gold_others).ratio()

    score = 0.5 * snippet_score + 0.5 * callsite_score
    if not executable or duplicate_defs or nesting_mismatch:
        score = min(score, 0.15)  # soft cap, see the no_gold_extraction branch above
    return {"score": score, "snippet_score": snippet_score,
            "callsite_score": callsite_score, "complexity": complexity,
            "executable": executable, "duplicate_defs": duplicate_defs,
            "reason": "nesting_mismatch" if nesting_mismatch else None}


# -- semantic (operator-aware) variants of extract_method/deduplicate --------
# Kept entirely separate from score_extract_method/score_deduplicate/
# _score_extract_method_multi above (never called by them, never calls them) so
# the currently in-flight run's scoring/RL-reward formula is untouched -- opt
# into these via score_extract_method_semantic/score_deduplicate_semantic/
# TASK_SCORERS_SEMANTIC below instead. Targets the documented blind spot in
# _score_extract_method_multi's docstring: a helper with one
# flipped comparison operator (`i > 0` -> `i < 0`, a completely different
# function) scored 0.996 under plain character-level SequenceMatcher, because
# the single differing character is diluted across the whole snippet's length.
_CRITICAL_OP_NODES = (_ast.Compare, _ast.BoolOp, _ast.BinOp, _ast.UnaryOp, _ast.AugAssign)


def _op_signature(text):
    """Ordered list of (node_type_name, op_type_name) for every critical
    semantic operator node (Compare/BoolOp/BinOp/UnaryOp/AugAssign) in text's
    AST, walked in ast.walk's pre-order. None on SyntaxError. A Compare node's
    chained comparisons (e.g. `0 < i < 10`) contribute one entry per op, in
    order. This is a short, low-noise sequence for a typical helper function
    body -- exactly the property that makes a single flipped operator visible
    as a real difference (e.g. 1/3 entries differ) instead of vanishing inside
    a long character-level ratio."""
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return None
    sig = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Compare):
            sig.extend(("Compare", type(op).__name__) for op in node.ops)
        elif isinstance(node, _CRITICAL_OP_NODES):
            sig.append((type(node).__name__, type(node.op).__name__))
    return sig


def _semantic_body_similarity(pred_body_text, gold_body_text):
    """Blends the existing character-level SequenceMatcher ratio (still needed
    to catch missing/extra logic, different variable usage, added statements
    -- things an operator-signature list alone says nothing about) with an
    explicit operator-signature match (see _op_signature) that specifically
    targets the flipped-operator blind spot the text ratio can't weight
    properly. Weighted 0.5/0.5 -- text ratio and op-signature ratio are
    independent signals, neither should dominate.

    Returns (score, op_match) -- op_match is exposed separately (mirrors how
    `executable`/`reason` surface other hard-to-see failure modes) so a low
    score can be attributed to "operators disagree" specifically, not just a
    generically lower blended number. op_match is None if either side fails to
    parse (caller already gets a snippet body from _extracted_helper_text, so
    this is not expected in practice, but handled defensively)."""
    text_ratio = SequenceMatcher(None, pred_body_text, gold_body_text).ratio()
    pred_sig = _op_signature(pred_body_text)
    gold_sig = _op_signature(gold_body_text)
    if pred_sig is None or gold_sig is None:
        return text_ratio, None
    op_ratio = 1.0 if not pred_sig and not gold_sig else \
        SequenceMatcher(None, pred_sig, gold_sig).ratio()
    return 0.5 * text_ratio + 0.5 * op_ratio, op_ratio


def _score_extract_method_multi_semantic(before_text, pred_text, gold_text,
                                          before_funcs, pred_funcs, gold_funcs):
    """Same shape/contract/early-return branches as _score_extract_method_multi
    (see its docstring for the full K==0/K>=2/executability-gate rationale --
    unchanged here), but the snippet-body comparison uses
    _semantic_body_similarity instead of a plain SequenceMatcher text ratio.
    New key: op_match -- the op-signature-only ratio for whichever pred helper
    ended up best-matched to each gold helper (min across matched pairs, so one
    badly-mismatched operator among several correctly-matched helpers still
    surfaces); None if there were no matched pairs at all."""
    gold_new = gold_funcs - before_funcs
    complexity = _gold_extract_method_n_lines(before_text, gold_text)
    executable = not _has_undefined_names(pred_text)

    if len(gold_new) == 0:
        s = SequenceMatcher(None, pred_text, gold_text).ratio()
        if not executable:
            s = min(s, 0.15)
        return {"score": s, "snippet_score": s, "callsite_score": s, "op_match": None,
                "complexity": complexity, "executable": executable,
                "reason": "no_gold_extraction"}

    gold_helpers = [_extracted_helper_text(gold_text, n) for n in gold_new]
    if any(h is None for h in gold_helpers):
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": complexity, "executable": executable,
                "reason": "extract_split_fail"}

    pred_new = pred_funcs - before_funcs
    if len(pred_new) > MAX_PRED_NEW_FUNCS:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": complexity, "executable": executable,
                "reason": "too_many_new_funcs"}
    pred_helpers = [h for h in (_extracted_helper_text(pred_text, n) for n in pred_new)
                    if h is not None]

    if not pred_helpers:
        snippet_score = 0.0
        op_match = None
    else:
        remaining = list(pred_helpers)
        matched, op_matches = [], []
        for gh in gold_helpers:
            best_i, best_r, best_op = -1, -1.0, None
            for i, ph in enumerate(remaining):
                r, op_r = _semantic_body_similarity(ph, gh)
                if r > best_r:
                    best_r, best_i, best_op = r, i, op_r
            matched.append(best_r if best_i >= 0 else 0.0)
            if best_i >= 0:
                if best_op is not None:
                    op_matches.append(best_op)
                remaining.pop(best_i)
        denom = max(len(gold_helpers), len(pred_helpers))
        snippet_score = sum(matched) / denom
        op_match = min(op_matches) if op_matches else None

    gold_others = _strip_named_funcs(gold_text, gold_new)
    pred_others = _strip_named_funcs(pred_text, pred_new)
    if gold_others is None or pred_others is None:
        callsite_score = 0.0
    else:
        callsite_score = SequenceMatcher(None, pred_others, gold_others).ratio()

    score = 0.5 * snippet_score + 0.5 * callsite_score
    if not executable:
        score = min(score, 0.15)
    return {"score": score, "snippet_score": snippet_score,
            "callsite_score": callsite_score, "op_match": op_match,
            "complexity": complexity, "executable": executable, "reason": None}


def score_extract_method_semantic(before_text, pred_text, gold_text):
    """Operator-signature-aware sibling of score_extract_method -- same call
    signature and every early-return branch, kept as a fully separate function
    (not a modification of score_extract_method) so the current run's
    scoring/RL-reward stays on the original formula. See
    _score_extract_method_multi_semantic / _semantic_body_similarity for the
    mechanism and new `op_match` key.
    """
    if pred_text is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    pred_text = pred_text.strip()
    gold_text = gold_text.strip()

    before_funcs = _func_names(before_text)
    pred_funcs = _func_names(pred_text)
    gold_funcs = _func_names(gold_text)
    if any(x is None for x in (before_funcs, pred_funcs, gold_funcs)):
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    gold_new = gold_funcs - before_funcs
    if len(gold_new) != 1:
        return _score_extract_method_multi_semantic(before_text, pred_text, gold_text,
                                                     before_funcs, pred_funcs, gold_funcs)
    gold_extracted_name = next(iter(gold_new))
    gold_norm = _rename_func(gold_text, gold_extracted_name, CANONICAL)
    gold_tree = _ast.parse(gold_norm)
    gold_extracted, gold_others = _split_extracted(gold_tree)
    if gold_extracted is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": None, "executable": False, "reason": "extract_split_fail"}
    n_lines = len(_ast.unparse(gold_extracted).splitlines())

    pred_new = pred_funcs - before_funcs
    if len(pred_new) != 1:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": n_lines, "executable": False, "reason": "wrong_new_func_count"}
    pred_extracted_name = next(iter(pred_new))
    pred_norm = _rename_func(pred_text, pred_extracted_name, CANONICAL)
    if pred_norm is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": n_lines, "executable": False, "reason": "parse_fail"}
    pred_tree = _ast.parse(pred_norm)
    pred_extracted, pred_others = _split_extracted(pred_tree)
    if pred_extracted is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": n_lines, "executable": False, "reason": "extract_split_fail"}

    executable = not _has_undefined_names(pred_text)

    snippet_score, op_match = _semantic_body_similarity(
        _ast.unparse(pred_extracted), _ast.unparse(gold_extracted)
    )
    pred_others_text = "\n".join(_ast.unparse(n) for n in pred_others)
    gold_others_text = "\n".join(_ast.unparse(n) for n in gold_others)
    callsite_score = SequenceMatcher(None, pred_others_text, gold_others_text).ratio()

    score = 0.5 * snippet_score + 0.5 * callsite_score
    if not executable:
        score = min(score, 0.15)
    return {"score": score, "snippet_score": snippet_score,
            "callsite_score": callsite_score, "op_match": op_match,
            "complexity": n_lines, "executable": executable, "reason": None}


def score_deduplicate_semantic(before_text, pred_text, gold_text):
    """Operator-signature-aware sibling of score_deduplicate -- same
    relationship to score_extract_method_semantic that score_deduplicate has to
    score_extract_method (reuses _score_extract_method_multi_semantic
    directly). Kept separate from score_deduplicate so the current run's
    scoring/RL-reward stays on the original formula."""
    if pred_text is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    pred_text = pred_text.strip()
    gold_text = gold_text.strip()

    before_funcs = _func_names(before_text)
    pred_funcs = _func_names(pred_text)
    gold_funcs = _func_names(gold_text)
    if any(x is None for x in (before_funcs, pred_funcs, gold_funcs)):
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0, "op_match": None,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    return _score_extract_method_multi_semantic(before_text, pred_text, gold_text,
                                                before_funcs, pred_funcs, gold_funcs)


def score_extract_method(before_text, pred_text, gold_text):
    """Continuous extract_method score, split into:
      snippet_score  -- does the extracted helper's body match gold's (name-agnostic)?
      callsite_score -- does the rest of the file (caller + call site) match gold's?
                         This is the "function calling correctness" check.
    Returns dict: score, snippet_score, callsite_score,
    complexity (n_lines = gold extracted-body line count), reason (on hard fail).
    """
    if pred_text is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    pred_text = pred_text.strip()
    gold_text = gold_text.strip()

    before_funcs = _func_names(before_text)
    pred_funcs = _func_names(pred_text)
    gold_funcs = _func_names(gold_text)
    if any(x is None for x in (before_funcs, pred_funcs, gold_funcs)):
        # Report the real gold-side complexity even on a pred/before/gold parse
        # failure -- n_lines is fully determined by before/gold and never
        # depended on pred parsing. Previously this returned complexity=None,
        # dumping every parse-failing pred into the eval's "n/a" complexity
        # bucket regardless of the (knowable) gold shape -- the same tautology
        # the _gold_* helpers already removed for rename/remove_dead_code.
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    gold_new = gold_funcs - before_funcs
    if len(gold_new) != 1:
        # Gold has 0, or 2+, new functions relative to before. The old code
        # fell back to `1.0 if pred_text == gold_text else 0.0` -- a near-
        # impossible exact-string match that read as ~total failure regardless
        # of real output quality. Replaced with a structural N-helper
        # comparison (see _score_extract_method_multi).
        return _score_extract_method_multi(before_text, pred_text, gold_text,
                                           before_funcs, pred_funcs, gold_funcs)
    gold_extracted_name = next(iter(gold_new))
    gold_norm = _rename_func(gold_text, gold_extracted_name, CANONICAL)
    gold_tree = _ast.parse(gold_norm)
    gold_extracted, gold_others = _split_extracted(gold_tree)
    if gold_extracted is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": None, "executable": False, "reason": "extract_split_fail"}
    n_lines = len(_ast.unparse(gold_extracted).splitlines())

    pred_new = pred_funcs - before_funcs
    if len(pred_new) != 1:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": n_lines, "executable": False, "reason": "wrong_new_func_count"}
    pred_extracted_name = next(iter(pred_new))
    pred_norm = _rename_func(pred_text, pred_extracted_name, CANONICAL)
    if pred_norm is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": n_lines, "executable": False, "reason": "parse_fail"}
    pred_tree = _ast.parse(pred_norm)
    pred_extracted, pred_others = _split_extracted(pred_tree)
    if pred_extracted is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": n_lines, "executable": False, "reason": "extract_split_fail"}

    # Executability gate, added alongside the deduplicate redesign (see
    # _score_extract_method_multi's docstring): a pred that extracts a
    # plausible-looking helper but forgets to pass a variable it needs (a real
    # NameError at the new call site) previously scored identically to a
    # correct extraction, since snippet_score/callsite_score are pure text
    # similarity and never checked this.
    executable = not _has_undefined_names(pred_text)
    # Duplicate-def gate : two defs sharing a name after
    # apply_sr, where the second silently shadows the first -- a real
    # correctness bug snippet/callsite text similarity alone can't see.
    duplicate_defs = bool(_duplicate_def_names(pred_text))
    # Nesting-shape gate : flag only when one side has the
    # helper genuinely nested inside another FUNCTION and the other doesn't --
    # that's always unreachable/broken regardless of callsite adaptation.
    # Module vs ClassDef is deliberately NOT flagged here: putting the new
    # helper at module level instead of as a class method (or vice versa) is
    # a legitimate structural choice this scorer is supposed to be tolerant
    # of (same name-agnostic/structure-flexible spirit as the rest of this
    # function) as long as callsite_score reflects a consistently-adapted
    # call site -- confirmed empirically: an adversarial case with a
    # correctly-adapted module-level helper (snippet_score=0.97,
    # callsite_score=0.89) was wrongly hard-capped to 0.15 by an earlier,
    # broader version of this check that flagged any Module/ClassDef diff.
    gold_scope_kind = _func_scope_kind(gold_text, gold_extracted_name)
    pred_scope_kind = _func_scope_kind(pred_text, pred_extracted_name)
    nesting_mismatch = (("FunctionDef" in (gold_scope_kind, pred_scope_kind)
                        and gold_scope_kind != pred_scope_kind)
                        or _existing_func_became_nested(before_text, pred_text))

    snippet_score = SequenceMatcher(
        None, _ast.unparse(pred_extracted), _ast.unparse(gold_extracted)
    ).ratio()
    pred_others_text = "\n".join(_ast.unparse(n) for n in pred_others)
    gold_others_text = "\n".join(_ast.unparse(n) for n in gold_others)
    callsite_score = SequenceMatcher(None, pred_others_text, gold_others_text).ratio()

    score = 0.5 * snippet_score + 0.5 * callsite_score
    if not executable or duplicate_defs or nesting_mismatch:
        score = min(score, 0.15)  # soft cap, see _score_extract_method_multi
    return {"score": score, "snippet_score": snippet_score,
            "callsite_score": callsite_score, "complexity": n_lines,
            "executable": executable, "duplicate_defs": duplicate_defs,
            "reason": "nesting_mismatch" if nesting_mismatch else None}


def extract_method_ok(before_text, pred_text, gold_text):
    """Backward-compatible binary wrapper around score_extract_method, used by
    the qualitative-sample cells below."""
    return score_extract_method(before_text, pred_text, gold_text)["score"] >= 0.999


# -- simplify / deduplicate: single canonical target, continuous similarity --
def _n_changed_hunks(before_text, gold_text):
    """# contiguous changed line-groups between before/gold -- generic complexity
    tag for tasks with no task-specific complexity notion of their own."""
    before_lines = before_text.strip("\n").split("\n")
    gold_lines = gold_text.strip("\n").split("\n")
    sm = SequenceMatcher(None, before_lines, gold_lines, autojunk=False)
    return sum(1 for tag, *_ in sm.get_opcodes() if tag != "equal")


_BUILTIN_NAMES = set(dir(_builtins)) | {"__class__"}
# "self"/"cls" deliberately NOT blanket-allowed here (removed later):
# before the scope-aware rewrite of _has_undefined_names, the old whole-
# snippet name check needed this as a workaround so a method's legitimate
# `self` reference wouldn't false-positive. Now that _scope_own_bound binds
# a function's own parameters (self/cls included) correctly per-scope, the
# blanket allowance is not just redundant, it actively HIDES real bugs: it
# treats `self` as bound EVERYWHERE, including a hunk-boundary corruption
# that dedents a statement out of its method entirely (sample #049 --
# `self.total *= 0.95` ends up at module scope, a genuine NameError at
# runtime, confirmed by the smoke test's real execution, but invisible to
# this gate as long as the blanket exception stood). `__class__` stays --
# Python implicitly binds it inside any method that calls zero-arg
# `super()`, which _scope_own_bound has no way to see (it's not a real
# Store-context binding in the AST).


def _collect_bound_names(tree):
    """All names that get a binding somewhere in the snippet: assignment/for/
    with/comprehension targets, function/class defs, parameters, imports,
    global/nonlocal declarations."""
    bound = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name) and isinstance(node.ctx, (_ast.Store, _ast.Del)):
            bound.add(node.id)
        elif isinstance(node, _ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, _ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, _ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, _ast.Global):
            bound.update(node.names)
        elif isinstance(node, _ast.Nonlocal):
            bound.update(node.names)
    return bound


def _scope_bound_locals(stmts):
    """Names bound directly within `stmts` (one scope's own statement list):
    assignment/for/with/comprehension targets, imports, global/nonlocal
    declarations, and the *names* of any nested def/class (which bind in the
    enclosing scope the same way a plain assignment would) -- but does NOT
    descend into a nested function/class/lambda's own body/params, which stay
    private to that nested scope. This is the piece that was missing before:
    the old _has_undefined_names walked the entire tree unconditionally via
    ast.walk, so a parameter belonging to one method looked "bound" from
    inside a completely different sibling method in the same class (sample
    #056: `order` is a real parameter of `process_order`/`refund_order`, but
    the rewritten `calculate_total(self, items)` uses `order` too -- undefined
    in that function's own scope, silently waved through because it was bound
    *somewhere* in the file)."""
    bound = set()

    class _Local(_ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, (_ast.Store, _ast.Del)):
                bound.add(node.id)
        def visit_FunctionDef(self, node):
            bound.add(node.name)
        def visit_AsyncFunctionDef(self, node):
            bound.add(node.name)
        def visit_ClassDef(self, node):
            bound.add(node.name)
        def visit_Lambda(self, node):
            return  # params are private to the lambda, no name bound outward
        def visit_Import(self, node):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        def visit_ImportFrom(self, node):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        def visit_Global(self, node):
            bound.update(node.names)
        def visit_Nonlocal(self, node):
            bound.update(node.names)

    visitor = _Local()
    for stmt in stmts:
        visitor.visit(stmt)
    return bound


def _func_arg_names(args):
    names = {a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _scope_own_bound(node):
    """Names bound directly within a scope-defining node's own body: a
    Module or ClassDef's top-level bindings, or a FunctionDef/
    AsyncFunctionDef's parameters plus its own body's top-level bindings.
    Used both to check names referenced within this scope and, via the
    parent-chain walk in _scope_visible_names, to build the correct closure
    for nested scopes."""
    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        return _func_arg_names(node.args) | _scope_bound_locals(node.body)
    return _scope_bound_locals(node.body)  # Module or ClassDef


def _build_parent_map(tree):
    parents = {}
    for node in _ast.walk(tree):
        for child in _ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _scope_visible_names(node, parents):
    """Names visible inside `node`'s own body purely from enclosing scopes:
    walks the parent chain, folding in each enclosing Module/FunctionDef/
    AsyncFunctionDef's own bound names (closures see their enclosing
    function/module normally) but SKIPPING ClassDef ancestors' own bound
    names entirely -- matching real Python scoping, where a class body's own
    names (its methods, class-level assignments) are NOT visible inside its
    own methods without going through self./ClassName. This asymmetry (class
    scope doesn't nest, function scope does) is exactly what the previous
    whole-snippet ast.walk collapsed away."""
    visible = set()
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef)):
            visible |= _scope_own_bound(current)
        current = parents.get(current)
    return visible


class _ShallowNameLoads(_ast.NodeVisitor):
    """Collects Load-context Name references reachable from a scope's own
    statements without crossing into a nested FunctionDef/AsyncFunctionDef/
    ClassDef/Lambda's private body -- those are each checked separately (as
    their own scope, with their own visible set) by the caller, not against
    this scope's names. Decorators, base classes, and default-value
    expressions on a nested def/class ARE evaluated in the enclosing scope in
    real Python, so those are still walked; only the body/params are skipped."""
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, _ast.Load):
            self.names.add(node.id)

    def visit_FunctionDef(self, node):
        for dec in node.decorator_list:
            self.visit(dec)
        for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)
        # Type annotations (arg annotations and the return annotation) ARE
        # evaluated in the enclosing scope at def time in real Python (absent
        # `from __future__ import annotations`) -- walked here too, otherwise
        # an undefined name used only in a type hint silently stops being
        # caught at all (a real coverage regression vs. the old whole-snippet
        # version, which happened to catch this by accident since it checked
        # every Name node in the tree with no scope distinction).
        for arg in (list(node.args.posonlyargs) + list(node.args.args)
                   + list(node.args.kwonlyargs)
                   + ([node.args.vararg] if node.args.vararg else [])
                   + ([node.args.kwarg] if node.args.kwarg else [])):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        # node.body is a separate scope -- not walked here, handled as its
        # own entry in _has_undefined_names' scope_nodes loop.

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw.value)
        # node.body is a separate scope -- not walked here.

    def visit_Lambda(self, node):
        for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)
        # A lambda's own expression body is its own tiny scope, not checked
        # separately -- heuristic tolerance (lambdas are rare in this corpus)
        # that only risks a false negative, same direction the old
        # whole-snippet check already erred in for every case.


def _has_undefined_names(code_text):
    """Heuristic executability check, catches the class of bug where a
    snippet is syntactically valid but references a name nothing in its
    actual (Python-scoping-correct) scope binds -- e.g. a loop variable whose
    defining `for` was deleted, or a name that's only a parameter on a
    *different* sibling method (sample #056) -- a real NameError at runtime
    that ast.parse alone can't see and that a naive whole-snippet name
    collection incorrectly waves through.

    Checks each scope (module top level, and every function/method/class
    body) independently against exactly the names Python would actually make
    visible there: its own bindings, plus closures from enclosing
    function/module scopes, plus builtins -- but NOT a sibling function's
    params/locals, and NOT a class's own body from inside its methods (see
    _scope_visible_names). Still snippet-local overall: a name only bound via
    real outer-scope context this fragment doesn't include (e.g. a module
    imported elsewhere in the real file) is still accepted as unknown-but-not
    ruled-out only through the builtins/enclosing-scope fallback -- this can
    still false-positive on legitimate outer-scope references, which is why
    every caller uses this as a soft penalty, not a hard fail."""
    try:
        tree = _ast.parse(code_text)
        # ast.parse alone is too permissive in CPython 3.13: it accepts
        # module-level `return` and `break`/`continue`/`yield`/`await` outside a
        # valid context, so a genuinely non-runnable snippet would slip through
        # as executable and dodge the low-score cap. compile() enforces those
        # context rules and raises SyntaxError, which this except catches.
        compile(tree, "<check>", "exec")
    except SyntaxError:
        return True

    parents = _build_parent_map(tree)
    scope_nodes = [tree] + [n for n in _ast.walk(tree)
                            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))]

    for scope in scope_nodes:
        own_bound = _scope_own_bound(scope)
        visible = _scope_visible_names(scope, parents) | own_bound | _BUILTIN_NAMES
        collector = _ShallowNameLoads()
        for stmt in scope.body:
            collector.visit(stmt)
        if any(name not in visible for name in collector.names):
            return True
    return False


def _duplicate_def_names(code_text):
    """Set of function/method names defined 2+ times at the *same* scope
    (module top level, or within the same class/function body) in
    code_text -- e.g. two methods in one class both named `process_order`
    after a bad SEARCH/REPLACE rewrite. The second definition silently
    shadows/overwrites the first at runtime (a whole method's worth of
    behavior is lost), which is a real correctness bug, but every scorer
    that reasons about renamed/extracted pieces in isolation previously had
    no way to notice it.
    Recurses per-scope (module body, each function's own body, each class's
    own body) so a name repeated across two *different* scopes (a top-level
    helper and an unrelated method sharing a name -- completely legal
    Python) is never flagged; only a same-scope collision counts. Returns an
    empty set on SyntaxError or when nothing collides.

    KNOWN FALSE-POSITIVE RISK: does not look at decorators, so a legitimate
    `@property`/`@x.setter`/`@x.deleter` trio (same method name, 2-3 times,
    by design) would be flagged too. Confirmed `@property` does occur at
    least a handful of times in this project's data (grep hit in
    supervised_finetuning_generated.jsonl and the curriculum/bridging
    corpora). Not special-cased here: every caller treats this as a soft
    cap (score<=0.15), not a hard fail, same tolerance-for-false-positives
    stance the rest of this module already takes for `_has_undefined_names`
    -- and property patterns are rare enough in this refactoring-snippet
    corpus that decorator-aware exclusion logic wasn't judged worth the
    added complexity yet. Revisit if a real eval run shows this misfiring
    in practice."""
    try:
        tree = _ast.parse(code_text)
    except SyntaxError:
        return set()

    dupes = set()

    def _check_scope(stmts):
        seen = {}
        for stmt in stmts:
            if isinstance(stmt, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                seen[stmt.name] = seen.get(stmt.name, 0) + 1
                _check_scope(stmt.body)
            elif isinstance(stmt, _ast.ClassDef):
                _check_scope(stmt.body)
        dupes.update(name for name, count in seen.items() if count > 1)

    _check_scope(tree.body)
    return dupes


def score_text_similarity(before_text, pred_text, gold_text):
    """Line-level similarity ratio against gold, with a soft executability
    penalty: a prediction that's textually close to gold but references an
    undefined name (e.g. a variable from a loop it just deleted) is real bug
    territory, not partial credit, so it's capped low instead of scoring on
    text closeness alone. deduplicate's known ambiguity (first-match apply_sr
    may patch a different, equally-valid duplicate occurrence than gold) shows
    up as a score modestly below 1.0 rather than a hard 0/1 miscount.
    Complexity tag: n_hunks (# changed line-groups)."""
    n_hunks = _n_changed_hunks(before_text, gold_text)
    if pred_text is None:
        return {"score": 0.0, "complexity": n_hunks, "executable": False, "reason": "parse_fail"}
    pred_text = pred_text.strip()
    gold_text = gold_text.strip()
    if pred_text == gold_text:
        return {"score": 1.0, "complexity": n_hunks, "executable": True, "reason": None}
    score = SequenceMatcher(None, pred_text, gold_text).ratio()
    executable = not _has_undefined_names(pred_text)
    if not executable:
        score = min(score, 0.15)
    return {"score": score, "complexity": n_hunks, "executable": executable, "reason": None}


# -- simplify: localization x gold-independent validity x non-interference --
def _line_opcodes(before_text, other_text):
    """(before_lines, other_lines, opcodes) for a before -> other line-level
    diff. Shared by every localization/non-interference computation below --
    same alignment _n_changed_hunks counts hunks from, just keeping the raw
    opcodes/lines instead of collapsing straight to a count."""
    before_lines = before_text.strip("\n").split("\n")
    other_lines = other_text.strip("\n").split("\n")
    opcodes = SequenceMatcher(None, before_lines, other_lines, autojunk=False).get_opcodes()
    return before_lines, other_lines, opcodes


def _changed_before_indices(opcodes):
    """Set of before-line positions a before->other diff's non-equal opcodes
    touch. A zero-width insert (i1 == i2, pure insertion with nothing deleted/
    replaced) contributes its anchor position i1 -- an insertion is "at" that
    point even though it claims no existing line; without this, an insert-only
    edit would look like it changed nothing at all."""
    changed = set()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        changed.add(i1) if i1 == i2 else changed.update(range(i1, i2))
    return changed


def _ast_node_count(text):
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return None
    return sum(1 for _ in _ast.walk(tree))


def _free_names(text):
    """Load-context Name references not bound anywhere in `text` and not a
    builtin -- i.e. the external dependencies (imported modules, globals) the
    snippet actually relies on. Reuses _collect_bound_names/_BUILTIN_NAMES
    (already built for _has_undefined_names) rather than a new binding pass."""
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return None
    bound = _collect_bound_names(tree) | _BUILTIN_NAMES
    return {node.id for node in _ast.walk(tree)
            if isinstance(node, _ast.Name) and isinstance(node.ctx, _ast.Load)
            and node.id not in bound}


def score_simplify(before_text, pred_text, gold_text):
    """Continuous simplify score: localization_recall x validity x
    non_interference, instead of raw text similarity to one canonical gold
    string (the old score_text_similarity behavior, still used for
    deduplicate's fallback path -- see TASK_SCORERS).

    Motivation: simplify's gold is one LLM-generated rewrite among many
    equally valid ones (the dataset builder's "simplify" task has no
    enumerable rule set behind it -- backward generation from a single clean
    target), so penalizing a different-but-correct simplification as if it
    were wrong systematically undercounts real quality. This scorer never
    requires pred's *text* to resemble gold's -- only that it touched the
    same region gold touched (localization_recall), left everything else
    alone (non_interference), and that whatever it put there is a real
    simplification on its own terms (validity), checked without needing gold
    at all:
      executable  -- _has_undefined_names(pred_text) is False (reused as-is).
      simplified  -- pred's whole-file AST node-count reduction, scaled
                      against how much gold itself reduced. Since apply_sr
                      guarantees everything outside the edited region is
                      byte-identical, the whole-file node-count delta *is*
                      the edited region's delta -- no need to re-parse an
                      isolated (and not-necessarily-valid-standalone-Python)
                      line slice. Continuous: clamp(pred_reduction /
                      gold_reduction, 0.0, 1.0), where pred_reduction =
                      before_nodes - pred_nodes and gold_reduction =
                      before_nodes - gold_nodes.

                      FIXED later (bug found earlier, deferred to the
                      TLM-M eval cycle): the old formula, min(1.0,
                      before_nodes/pred_nodes), capped at full credit for ANY
                      node-count reduction, however trivial, with no
                      reference to how much gold itself reduced -- `pred =
                      "return x + 0 - 0"` against `gold = "return x"` scored
                      simplified=1.0 despite keeping the redundant arithmetic
                      gold fully removed. Now scaled against gold's own
                      reduction instead. `gold_reduction <= 0` (gold itself
                      didn't reduce node count -- degenerate/rare, the
                      dataset generator rejects before==after but doesn't
                      guarantee a strict reduction) falls back to the old
                      any-reduction-gets-credit formula rather than dividing
                      by a non-positive denominator. BREAKING CHANGE: this
                      scorer also backs rl_training.py's reward function --
                      any score from before this fix is not comparable to a
                      post-fix re-eval; re-baseline before comparing runs
                      across this change.
      io_preserved -- Jaccard similarity of _free_names(before) vs
                      _free_names(pred): a valid simplify shouldn't start
                      depending on a new global/import it didn't have before.

    Returns dict: score, localization_recall, validity, non_interference,
    executable, complexity (n_hunks, before/gold), reason (on hard fail).
    """
    before_lines, gold_lines, gold_opcodes = _line_opcodes(before_text, gold_text)
    gold_changed = _changed_before_indices(gold_opcodes)
    n_hunks = _n_changed_hunks(before_text, gold_text)

    if pred_text is None:
        return {"score": 0.0, "localization_recall": 0.0, "validity": 0.0,
                "non_interference": 0.0, "executable": False,
                "complexity": n_hunks, "reason": "parse_fail"}

    pred_text = pred_text.strip()
    gold_text = gold_text.strip()
    if pred_text == gold_text:
        return {"score": 1.0, "localization_recall": 1.0, "validity": 1.0,
                "non_interference": 1.0, "executable": True,
                "complexity": n_hunks, "reason": None}

    _, pred_lines, pred_opcodes = _line_opcodes(before_text, pred_text)
    pred_changed = _changed_before_indices(pred_opcodes)

    if not gold_changed:
        # gold's own before->gold diff is empty (degenerate -- the generator
        # rejects before==after, but a whitespace-only diff could still land
        # here) -- nothing to localize; score purely on whether pred also
        # left before alone, same edge-case shape as score_rename's n_sites==0.
        non_interference = 1.0 - len(pred_changed) / max(len(before_lines), 1)
        return {"score": non_interference, "localization_recall": 1.0,
                "validity": 1.0, "non_interference": non_interference,
                "executable": True, "complexity": 0, "reason": None}

    localization_recall = len(gold_changed & pred_changed) / len(gold_changed)

    unchanged_before = set(range(len(before_lines))) - gold_changed
    if unchanged_before:
        interfered = len(pred_changed & unchanged_before)
        non_interference = 1.0 - interfered / len(unchanged_before)
    else:
        non_interference = 1.0

    executable = not _has_undefined_names(pred_text)

    before_nodes = _ast_node_count(before_text)
    pred_nodes = _ast_node_count(pred_text)
    gold_nodes = _ast_node_count(gold_text)
    if before_nodes and pred_nodes and gold_nodes:
        gold_reduction = before_nodes - gold_nodes
        if gold_reduction > 0:
            pred_reduction = before_nodes - pred_nodes
            simplified = max(0.0, min(1.0, pred_reduction / gold_reduction))
        else:
            # gold itself didn't reduce node count -- can't meaningfully
            # scale against a non-positive denominator, fall back to the
            # pre-fix any-reduction-gets-credit formula for this edge case.
            simplified = min(1.0, before_nodes / pred_nodes)
    else:
        simplified = 0.0

    before_free = _free_names(before_text)
    pred_free = _free_names(pred_text)
    if before_free is not None and pred_free is not None:
        union = before_free | pred_free
        io_preserved = len(before_free & pred_free) / len(union) if union else 1.0
    else:
        io_preserved = 0.0

    validity = 0.5 * simplified + 0.5 * io_preserved
    score = localization_recall * validity * non_interference
    if not executable:
        # Soft cap, not a hard zero -- see _score_extract_method_multi's
        # docstring: _has_undefined_names false-positives on legitimate
        # outer-scope references at a real, non-negligible rate (confirmed
        # empirically against this same dataset), so a hard 0/1 gate here
        # would have reintroduced exactly the discontinuity this redesign
        # was meant to remove. Same soft-cap value as score_text_similarity.
        score = min(score, 0.15)

    return {"score": score, "localization_recall": localization_recall,
            "validity": validity, "non_interference": non_interference,
            "executable": executable, "complexity": n_hunks, "reason": None}


# -- deduplicate: reuses extract_method's name-agnostic multi-helper match ---
def score_deduplicate(before_text, pred_text, gold_text):
    """Continuous deduplicate score. deduplicate's AFTER shape (per
    the dataset builder's task definition: duplicated logic at 2+ call
    sites in before -> one shared function, called from each site in after)
    is structurally the same thing extract_method's K>=2 path already scores:
    "some new functions appeared; the rest of the file changed to call them
    instead of repeating logic inline." Reuses _score_extract_method_multi
    directly rather than reinventing name-agnostic helper matching + remainder
    comparison -- lower risk than a bespoke metric, and it already gives
    partial credit for a differently-shaped (but still valid) consolidation
    than gold's specific one: it never requires pred's helper to match gold's
    helper by name, only that its body plausibly matches (best-match,
    name-agnostic via CANONICAL renaming) and that every place gold delegated,
    pred's remainder also reads as delegating. Also inherits the executability
    gate added there in the same session (a helper that references a variable
    never passed to it is a real bug, not a scoring nuance).

    Returns dict: score, snippet_score, callsite_score, complexity (n_lines,
    the gold new-function(s) total body line count), executable,
    reason (on hard fail).
    """
    if pred_text is None:
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    pred_text = pred_text.strip()
    gold_text = gold_text.strip()

    before_funcs = _func_names(before_text)
    pred_funcs = _func_names(pred_text)
    gold_funcs = _func_names(gold_text)
    if any(x is None for x in (before_funcs, pred_funcs, gold_funcs)):
        return {"score": 0.0, "snippet_score": 0.0, "callsite_score": 0.0,
                "complexity": _gold_extract_method_n_lines(before_text, gold_text),
                "executable": False, "reason": "parse_fail"}

    return _score_extract_method_multi(before_text, pred_text, gold_text,
                                        before_funcs, pred_funcs, gold_funcs)


# -- remove_dead_code: recall x precision x non-interference -----------------
def score_remove_dead_code(before_text, pred_text, gold_text):
    """recall (all gold-marked dead lines actually removed) x precision (no
    live lines removed) x non_interference (surviving lines left otherwise
    byte-identical). `before_text` must already be decoded/round-tripped
    through the same tokenizer as pred_text/gold_text (the caller is
    responsible for this) -- comparing raw source text against decoded text
    here previously counted pure formatting differences (tokenizer strips
    spacing around `=`/`,` etc.) as spurious non_interference failures even
    when the model's edit was exactly right. Complexity tag: n_dead_lines (gold).

    n_dead_lines is computed from before/gold alone, before the pred_text is
    None check -- it never depended on pred parsing, so a total apply_sr
    failure no longer discards complexity info that was already known (see
    score_rename's _gold_rename_complexity for the same fix and why it
    matters for the eval notebook's complexity-stratified "n/a" buckets)."""
    before_lines = before_text.strip("\n").split("\n")
    gold_lines = gold_text.strip().split("\n")

    sm_gold = SequenceMatcher(None, before_lines, gold_lines, autojunk=False)
    gold_removed = set()
    for tag, i1, i2, j1, j2 in sm_gold.get_opcodes():
        if tag in ("delete", "replace"):
            gold_removed.update(range(i1, i2))
    n_dead_lines = len(gold_removed)

    if pred_text is None:
        return {"score": 0.0, "recall": 0.0, "precision": 0.0,
                "non_interference": 0.0, "complexity": n_dead_lines, "reason": "parse_fail"}

    pred_lines = pred_text.strip().split("\n")

    sm_pred = SequenceMatcher(None, before_lines, pred_lines, autojunk=False)
    pred_removed = set()
    for tag, i1, i2, j1, j2 in sm_pred.get_opcodes():
        if tag in ("delete", "replace"):
            pred_removed.update(range(i1, i2))

    if n_dead_lines == 0:
        recall = 1.0
        precision = 1.0 if not pred_removed else 0.0
    else:
        recall = len(gold_removed & pred_removed) / n_dead_lines
        precision = (len(gold_removed & pred_removed) / len(pred_removed)
                     if pred_removed else (1.0 if not gold_removed else 0.0))

    kept_before = [ln for i, ln in enumerate(before_lines) if i not in gold_removed]
    kept_pred_text = "\n".join(pred_lines)
    kept_before_text = "\n".join(kept_before)
    non_interference = SequenceMatcher(None, kept_pred_text, kept_before_text).ratio()

    score = recall * precision * non_interference
    return {"score": score, "recall": recall, "precision": precision,
            "non_interference": non_interference, "complexity": n_dead_lines,
            "reason": None}


TASK_SCORERS = {
    "rename": score_rename,
    "extract_method": score_extract_method,
    "simplify": score_simplify,
    "deduplicate": score_deduplicate,
    "remove_dead_code": score_remove_dead_code,
}

# Canonical applied_ok cutoff (is_ok = score >= threshold), same convention the
# SFT eval notebook (score_eval.ipynb) uses. rename/remove_dead_code's
# scores are recall/consistency/precision products over discrete sites/lines -- many
# different genuinely-correct outputs land exactly at 1.0, so >=0.999 is a meaningful
# "fully correct" bar there. simplify/deduplicate/extract_method's scores are built
# from SequenceMatcher ratios and continuous AST-based proxies that essentially never
# land at exact 1.0 unless pred is byte-identical to gold, so 0.85 is used instead of
# a near-exact-match bar. Defined here (not just in the notebook) so training code
# (e.g. confidence_head_training.py) can use the SAME real-scorer-based "ok" verdict
# instead of inventing a separate, looser notion of correctness.
OK_SCORE_THRESHOLD = {
    "rename": 0.999,
    "remove_dead_code": 0.999,
    "simplify": 0.85,
    "deduplicate": 0.85,
    "extract_method": 0.85,
}

# Opt-in variant: identical to TASK_SCORERS except extract_method/deduplicate
# use the operator-signature-aware scorers above. rename/simplify/remove_dead_code
# are the exact same functions -- untouched by this change. Use this dict (not
# TASK_SCORERS) in a new eval pass or RL run that wants the semantic-aware
# scoring; the current in-flight run keeps using TASK_SCORERS unmodified.
TASK_SCORERS_SEMANTIC = {
    "rename": score_rename,
    "extract_method": score_extract_method_semantic,
    "simplify": score_simplify,
    "deduplicate": score_deduplicate_semantic,
    "remove_dead_code": score_remove_dead_code,
}
