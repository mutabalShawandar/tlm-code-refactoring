# -- Bridging: synthetic task-conditioned SR examples from the pretrain corpus --
#
# Teaches task-token conditioning and (eventually) insertion-style edits before
# SFT sees the curated real-world dataset -- neither is exercised by pretrain's
# span-corruption objective (see the project spec).
#
# Design note (found the hard way): AST source line
# numbers CANNOT be used as indices into `split_lines_by_newline`'s tokenized
# output -- the tokenizer injects synthetic dedent-only trailing lines and
# normalizes whitespace, so text-line-count != tokenized-line-count. Two safe
# patterns are used here instead, never mixed:
#   1. rename (make_rename_bridge): in-place identifier substitution has to
#      happen in BPE-id space (reusing make_sr_corruption's pretokenize/BPE-
#      range infrastructure), because it edits *within* an existing tokenized
#      line and needs to hand the result to apply_sr's own line indexing.
#   2. Pure text insertion/deletion tasks (remove_dead_code and others below):
#      build `before`/`after` as plain text via AST *character offsets into
#      the raw source string* (safe -- this never touches tokenized-line
#      space), then hand the {before, after} text pair to the already-proven
#      make_sft_sr/diff_hunks pipeline, which does its own from-scratch
#      tokenization of each side. No index space is ever shared/mixed across
#      the AST-text and BPE-id worlds within a single generator.

import ast
import builtins
import random
from typing import Dict, List, Optional, Tuple

from .pretrain_loaders import (
    pretokenize_with_docstring_mask,
    pretoken_to_bpe_ranges,
)
from .sft_corruption import (
    TASK_ID, TASK_TOKEN_NAME,
    split_lines_by_newline, join_lines, apply_sr, make_sft_sr,
)

_BUILTIN_NAMES = set(dir(builtins))
_RESERVED = {"self", "cls"}
_DEGRADED_POOL = ["x", "y", "z", "a", "b", "c", "tmp", "val", "v1", "v2", "obj", "d", "e", "n", "m"]


def _collect_local_identifiers(tree: ast.AST) -> List[str]:
    """Local variable/param/function names eligible for renaming: candidate
    *names* only (strings) -- never positions. Excludes builtins, self/cls,
    and imported names."""
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])

    candidates = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name not in _BUILTIN_NAMES and node.name not in imported:
                candidates.add(node.name)
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg not in _RESERVED and arg.arg not in _BUILTIN_NAMES:
                    candidates.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if (node.id not in _RESERVED and node.id not in _BUILTIN_NAMES
                    and node.id not in imported):
                candidates.add(node.id)
    return sorted(candidates)


def _build_line_map(bpe_ids: List[int], newline_id: int) -> List[int]:
    """Same as span_selection_and_corruption._build_line_map: BPE position -> line index."""
    line_map = []
    cur = 0
    for tid in bpe_ids:
        line_map.append(cur)
        if tid == newline_id:
            cur += 1
    return line_map


def make_rename_bridge(
    code: str, tok, cfg, rng: Optional[random.Random] = None,
) -> Optional[Dict]:
    """Build one rename bridging example from real (uncorrupted) code.

    Returns {type: "rename", before, after, spans} or None if unsuitable
    (parse error, no renameable identifiers, or fails self-validation).
    `after` == the real code unchanged; `before` has 1-3 local identifiers
    consistently replaced by a degraded name at every occurrence.

    All location/substitution work happens in BPE-id space (never AST
    line/col into text) -- see module docstring. Self-validates via the same
    `apply_sr` the curated dataset round-trips through before returning,
    rejecting (returning None) rather than emitting a bad record."""
    rng = rng or random
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    candidates = _collect_local_identifiers(tree)
    if not candidates:
        return None

    NEWLINE_ID = cfg.special_token_ids["newline"]
    BOS = cfg.special_token_ids["bos"]
    EOS = cfg.special_token_ids["eos"]
    SEARCH = cfg.special_token_ids["search"]
    REPLACE = cfg.special_token_ids["replace"]

    pretokens, in_ds = pretokenize_with_docstring_mask(tok, code)
    if not pretokens:
        return None
    pt_ranges = pretoken_to_bpe_ranges(tok, pretokens)

    name_to_pt_idxs: Dict[str, List[int]] = {}
    for i, pt in enumerate(pretokens):
        if in_ds[i]:
            continue
        if pt in candidates:
            name_to_pt_idxs.setdefault(pt, []).append(i)

    usable = [n for n in candidates if name_to_pt_idxs.get(n)]
    if not usable:
        return None

    n_pick = min(len(usable), rng.choice([1, 2, 2, 3]))   # bias toward 2, per brief
    chosen = rng.sample(usable, n_pick)

    existing_names = set(pretokens)
    used_degraded = set()
    rename_map: Dict[str, str] = {}
    pool = _DEGRADED_POOL[:]
    rng.shuffle(pool)
    for name in chosen:
        degraded = next((c for c in pool if c not in existing_names and c not in used_degraded), None)
        if degraded is None:
            degraded = f"tmp{len(used_degraded)}"
        used_degraded.add(degraded)
        rename_map[name] = degraded

    content_ids = tok.encode(code)         # == "after" ids
    N = len(content_ids)
    if N == 0:
        return None

    replacements: List[Tuple[int, int, List[int]]] = []   # (start, end, degraded_ids)
    for name, degraded in rename_map.items():

        degraded_ids = tok.tokenizer.encode(degraded).ids
        if not degraded_ids:
            continue
        for pt_idx in name_to_pt_idxs[name]:
            start, end = pt_ranges[pt_idx]
            if end <= start or end > N:
                continue
            replacements.append((start, end, degraded_ids))

    if not replacements:
        return None
    replacements.sort(key=lambda t: t[0])
    for k in range(1, len(replacements)):
        if replacements[k][0] < replacements[k - 1][1]:
            return None                     # overlapping spans; skip, don't guess

    # build corrupted "before" ids (single pass splice, mirrors make_sr_corruption)
    before_ids: List[int] = []
    prev = 0
    for start, end, degraded_ids in replacements:
        before_ids.extend(content_ids[prev:start])
        before_ids.extend(degraded_ids)
        prev = end
    before_ids.extend(content_ids[prev:])

    line_map = _build_line_map(content_ids, NEWLINE_ID)
    touched_lines = sorted({line_map[start] for start, _, _ in replacements})

    before_lines = split_lines_by_newline(before_ids, NEWLINE_ID)
    after_lines = split_lines_by_newline(content_ids, NEWLINE_ID)
    if len(before_lines) != len(after_lines):
        return None                         # substitution shouldn't change line count; be defensive anyway

    dec_body: List[int] = []
    spans_out = []
    for idx in touched_lines:
        if idx >= len(before_lines) or idx >= len(after_lines):
            return None
        search_ids = before_lines[idx]
        replace_ids = after_lines[idx]
        dec_body += [SEARCH] + search_ids
        dec_body += [REPLACE] + replace_ids
        spans_out.append({
            "line": idx,
            "search": tok.decode(search_ids),
            "replace": tok.decode(replace_ids),
        })
    dec_ids = [BOS] + dec_body + [EOS]

    before_text = tok.decode(before_ids)
    after_text = tok.decode(content_ids)

    record = {
        "type": "rename",
        "before": before_text,
        "after": after_text,
        "spans": spans_out,
    }

    # -- self-validate before returning (reject, don't guess) --------------
    # Both must parse as valid Python -- the round-trip check below only
    # proves token-level reconstruction, not that the reconstructed text is
    # even syntactically valid (see _validate_text_record's docstring for
    # the full scope/limits of what round-trip validation does and doesn't
    # prove; the same caveat applies here).
    try:
        ast.parse(before_text)
        ast.parse(after_text)
    except (SyntaxError, ValueError):
        return None

    result = make_bridge_sr(record, tok, cfg)
    if result is None:
        return None
    _, dec_ids_check, _ = result
    before_ids_check = tok.encode(before_text)
    after_ids_check = tok.encode(after_text)
    pred = apply_sr(before_ids_check, dec_ids_check, cfg)
    gold = join_lines(split_lines_by_newline(after_ids_check, NEWLINE_ID), NEWLINE_ID)
    if pred != gold:
        return None

    return record


def make_bridge_sr(
    record: Dict, tok, cfg, max_src_len: int = 512, max_tgt_len: int = 512,
) -> Optional[Tuple[List[int], List[int], int]]:
    """Build (enc_ids, dec_ids, task_id) from a bridging record's own tracked
    `spans` (line indices), instead of diff_hunks -- the generator already
    knows exactly which lines it edited, so there's no SequenceMatcher/
    ambiguity risk to inherit from sft_corruption.py's curated-dataset path.

    Returns None if the task type is unknown, `before` is empty, a span's
    line index is out of range for the current tokenization of before/after,
    or either sequence exceeds its length budget."""
    ttype = record.get("type")
    if ttype not in TASK_ID:
        return None

    BOS = cfg.special_token_ids["bos"]
    EOS = cfg.special_token_ids["eos"]
    NL = cfg.special_token_ids["newline"]
    SEARCH = cfg.special_token_ids["search"]
    REPLACE = cfg.special_token_ids["replace"]
    TASK_TOK = cfg.special_token_ids[TASK_TOKEN_NAME[ttype]]

    before_ids = tok.encode(record["before"])
    after_ids = tok.encode(record["after"])
    if not before_ids:
        return None

    before_lines = split_lines_by_newline(before_ids, NL)
    after_lines = split_lines_by_newline(after_ids, NL)

    dec_body: List[int] = []
    for span in record.get("spans", []):
        idx = span["line"]
        if idx >= len(before_lines) or idx >= len(after_lines):
            return None
        dec_body += [SEARCH] + before_lines[idx]
        dec_body += [REPLACE] + after_lines[idx]
    if not dec_body:
        return None

    enc_ids = [BOS, TASK_TOK] + before_ids + [EOS]
    dec_ids = [BOS] + dec_body + [EOS]

    if len(enc_ids) > max_src_len or len(dec_ids) > max_tgt_len:
        return None

    return enc_ids, dec_ids, TASK_ID[ttype]


def _validate_text_record(record: Dict, tok, cfg) -> bool:
    """Shared self-validation for text-pair generators (pattern 2 in the
    module docstring): (1) both `before` and `after` must themselves be
    syntactically valid Python -- the SR round-trip check below only proves
    the decoder target *reconstructs the after string*, it says nothing
    about whether that string (or `before`) is valid Python at all; a
    transform bug could in principle produce garbled-but-still-round-
    trippable text, since apply_sr operates purely on tokens, not on Python
    semantics. (2) build via make_sft_sr, apply_sr, compare to `after`.
    Reject (return False) rather than guess -- same discipline throughout
    this module and consistent with round_trip_check.py's philosophy.

    Note the scope of what this does NOT prove: that `after` is *behaviorally
    equivalent* to `before` when executed (e.g. that extract_method's helper
    truly receives every variable it needs, or that a boolop/ternary
    expansion preserves short-circuit side effects for expressions that have
    them). That is currently argued by construction per-transform (each
    rule is a standard, well-known equivalence, or backed by the free-
    variable-availability analysis for extract_method), not proven by
    executing the code -- executing arbitrary untrusted corpus snippets to
    check behavioral equivalence would be its own safety liability and is
    deliberately not attempted here."""
    try:
        ast.parse(record["before"])
        ast.parse(record["after"])
    except (SyntaxError, ValueError):
        return False

    NEWLINE_ID = cfg.special_token_ids["newline"]
    result = make_sft_sr(record, tok, cfg)
    if result is None:
        return False
    _, dec_ids_check, _ = result
    before_ids_check = tok.encode(record["before"])
    after_ids_check = tok.encode(record["after"])
    pred = apply_sr(before_ids_check, dec_ids_check, cfg)
    gold = join_lines(split_lines_by_newline(after_ids_check, NEWLINE_ID), NEWLINE_ID)
    return pred == gold


_DEAD_VAR_POOL = ["_dead", "_unused", "_tmp_dead", "_scratch", "_leftover"]
_UNUSED_IMPORT_POOL = ["os", "sys", "re", "json", "math", "itertools", "collections"]


def _unreachable_after_return_or_raise(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Insert one dead line immediately after a real return/raise statement,
    at the same indentation -- unreachable regardless of what originally
    followed, so no control-flow analysis is needed beyond locating the node.
    Uses AST character offsets into the raw source text (never tokenized-line
    indices) to build the corrupted text; see module docstring pattern 2."""
    candidates = [n for n in ast.walk(tree) if isinstance(n, (ast.Return, ast.Raise))]
    if not candidates:
        return None
    node = rng.choice(candidates)
    indent = " " * node.col_offset
    dead_name = rng.choice(_DEAD_VAR_POOL)
    template = rng.choice([
        f"{indent}{dead_name} = 0",
        f"{indent}pass",
        f"{indent}{dead_name} = None",
    ])
    lines = code.split("\n")
    insert_at = node.end_lineno              # 1-indexed end line -> insert right after it
    if insert_at > len(lines):
        return None
    lines.insert(insert_at, template)
    return "\n".join(lines)


def _duplicate_or_unused_import(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Insert a duplicate of an existing import, or a fresh unused import if
    there are none, as a new top-level line. Module-level statements are
    unindented, so this never needs to compute an indentation prefix."""
    existing_imports = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    lines = code.split("\n")
    if existing_imports:
        node = rng.choice(existing_imports)
        dup_text = "\n".join(lines[node.lineno - 1:node.end_lineno])
        insert_at = node.end_lineno
        lines.insert(insert_at, dup_text)
    else:
        already = {
            n.names[0].name for n in ast.walk(tree)
            if isinstance(n, ast.Import) and n.names
        }
        pool = [m for m in _UNUSED_IMPORT_POOL if m not in already]
        if not pool:
            return None
        mod = rng.choice(pool)
        lines.insert(0, f"import {mod}")
    return "\n".join(lines)


def make_remove_dead_code_bridge(
    code: str, tok, cfg, rng: Optional[random.Random] = None,
) -> Optional[Dict]:
    """Build one remove_dead_code bridging example from real (clean) code.

    Returns {type: "remove_dead_code", before, after} or None if unsuitable.
    `after` == the real code unchanged; `before` has one dead-code line
    injected (unreachable-after-return/raise, or a duplicate/unused import).

    Unlike rename, this builds `before`/`after` as plain text (safe here --
    see module docstring pattern 2) and reuses the already-validated
    make_sft_sr/diff_hunks pipeline for serialization, rather than
    reimplementing line-index tracking. Self-validates via apply_sr before
    returning, same discipline as make_rename_bridge."""
    rng = rng or random
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    generators = [_unreachable_after_return_or_raise, _duplicate_or_unused_import]
    rng.shuffle(generators)
    before_text = None
    for gen in generators:
        before_text = gen(code, tree, rng)
        if before_text is not None:
            break
    if before_text is None or before_text == code:
        return None

    record = {"type": "remove_dead_code", "before": before_text, "after": code}
    if not _validate_text_record(record, tok, cfg):
        return None
    return record


def _replace_source_lines(code: str, first_lineno: int, last_lineno: int, new_lines: List[str]) -> str:
    """Splice `new_lines` in place of source lines [first_lineno, last_lineno]
    (1-indexed, inclusive). Plain text list-slice replacement -- see module
    docstring pattern 2."""
    lines = code.split("\n")
    return "\n".join(lines[:first_lineno - 1] + new_lines + lines[last_lineno:])


def _listcomp_to_loop(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Expand `target = [expr for var in it if cond]` into an explicit
    for-loop + append. Only single-generator, at-most-one-`if` comprehensions
    assigned to a single target on one physical source line -- multi-clause/
    multi-line comprehensions are skipped rather than guessed at."""
    candidates = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.value, ast.ListComp)
                and node.lineno == node.end_lineno):
            comp = node.value
            if len(comp.generators) == 1 and not comp.generators[0].is_async \
                    and len(comp.generators[0].ifs) <= 1:
                candidates.append(node)
    if not candidates:
        return None

    node = rng.choice(candidates)
    comp = node.value
    gen = comp.generators[0]
    indent = " " * node.col_offset

    target_text = ast.get_source_segment(code, node.targets[0])
    var_text = ast.get_source_segment(code, gen.target)
    iter_text = ast.get_source_segment(code, gen.iter)
    expr_text = ast.get_source_segment(code, comp.elt)
    if None in (target_text, var_text, iter_text, expr_text):
        return None

    new_lines = [f"{indent}{target_text} = []", f"{indent}for {var_text} in {iter_text}:"]
    if gen.ifs:
        cond_text = ast.get_source_segment(code, gen.ifs[0])
        if cond_text is None:
            return None
        new_lines.append(f"{indent}    if {cond_text}:")
        new_lines.append(f"{indent}        {target_text}.append({expr_text})")
    else:
        new_lines.append(f"{indent}    {target_text}.append({expr_text})")

    return _replace_source_lines(code, node.lineno, node.end_lineno, new_lines)


def _ternary_return_to_ifelse(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Expand `return A if cond else B` into an if/else block. Only single
    physical source line ternary returns -- multi-line ones are skipped."""
    candidates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.IfExp)
        and n.lineno == n.end_lineno
    ]
    if not candidates:
        return None

    node = rng.choice(candidates)
    ifexp = node.value
    indent = " " * node.col_offset

    body_text = ast.get_source_segment(code, ifexp.body)
    cond_text = ast.get_source_segment(code, ifexp.test)
    orelse_text = ast.get_source_segment(code, ifexp.orelse)
    if None in (body_text, cond_text, orelse_text):
        return None

    new_lines = [
        f"{indent}if {cond_text}:",
        f"{indent}    return {body_text}",
        f"{indent}else:",
        f"{indent}    return {orelse_text}",
    ]
    return _replace_source_lines(code, node.lineno, node.end_lineno, new_lines)


_AUGOP_SYMBOLS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>",
    ast.BitOr: "|", ast.BitXor: "^", ast.BitAnd: "&", ast.MatMult: "@",
}


def _augassign_to_expanded(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Expand `target OP= value` into `target = target OP (value)`. Only
    single physical source line augmented assignments with a known operator
    symbol -- `value` is parenthesized to preserve precedence regardless of
    what it contains."""
    candidates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AugAssign) and n.lineno == n.end_lineno
        and type(n.op) in _AUGOP_SYMBOLS
    ]
    if not candidates:
        return None

    node = rng.choice(candidates)
    indent = " " * node.col_offset
    target_text = ast.get_source_segment(code, node.target)
    value_text = ast.get_source_segment(code, node.value)
    if None in (target_text, value_text):
        return None

    op_sym = _AUGOP_SYMBOLS[type(node.op)]
    new_line = f"{indent}{target_text} = {target_text} {op_sym} ({value_text})"
    return _replace_source_lines(code, node.lineno, node.end_lineno, [new_line])


def _boolop_return_to_ifelse(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Expand `return A or B` / `return A and B` into an if/else block. Only
    single physical source line, exactly-two-operand boolean returns --
    mirrors the ternary expansion's scope restriction. Like the ternary case,
    this evaluates A twice (once in the condition, once in the return) --
    a mechanical, not perf/side-effect-preserving, expansion, matching the
    original bridging brief's own description of this idiom."""
    candidates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.BoolOp)
        and len(n.value.values) == 2 and n.lineno == n.end_lineno
    ]
    if not candidates:
        return None

    node = rng.choice(candidates)
    boolop = node.value
    indent = " " * node.col_offset
    a_text = ast.get_source_segment(code, boolop.values[0])
    b_text = ast.get_source_segment(code, boolop.values[1])
    if None in (a_text, b_text):
        return None

    cond_text = a_text if isinstance(boolop.op, ast.Or) else f"not ({a_text})"
    new_lines = [
        f"{indent}if {cond_text}:",
        f"{indent}    return {a_text}",
        f"{indent}else:",
        f"{indent}    return {b_text}",
    ]
    return _replace_source_lines(code, node.lineno, node.end_lineno, new_lines)


def _and_if_to_nested(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Expand `if A and B:` into nested `if A:\\n    if B:`
    an AST-detector audit of the real curated simplify dataset found this is
    the single largest uncovered pattern -- 22.0% of real records have a
    mergeable nested-if in `before`, vs. 0% coverage from the original 4
    generators. Only a top-level 2-operand `and` in
    an `if` test with no elif/else on the statement (an else's semantics
    don't survive the split unmodified -- which branch would it attach to?),
    single physical line test only, so the textual splice stays unambiguous.
    Assumes the file's own indent step is 4 spaces to re-indent the body one
    level deeper, matching every other generator in this module (none of
    them compute the step dynamically either)."""
    candidates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and not n.orelse
        and isinstance(n.test, ast.BoolOp) and isinstance(n.test.op, ast.And)
        and len(n.test.values) == 2 and n.test.lineno == n.test.end_lineno
    ]
    if not candidates:
        return None

    node = rng.choice(candidates)
    a_text = ast.get_source_segment(code, node.test.values[0])
    b_text = ast.get_source_segment(code, node.test.values[1])
    if None in (a_text, b_text):
        return None

    indent = " " * node.col_offset
    body_lines = code.split("\n")[node.body[0].lineno - 1: node.body[-1].end_lineno]
    reindented = [(f"    {line}" if line.strip() else line) for line in body_lines]

    new_lines = [f"{indent}if {a_text}:", f"{indent}    if {b_text}:"] + reindented
    return _replace_source_lines(code, node.lineno, node.end_lineno, new_lines)


def _plain_bool_to_redundant_compare(code: str, tree: ast.AST, rng: random.Random) -> Optional[str]:
    """Expand `if x:` -> `if x == True:` and `if not x:` -> `if x == False:`.
    Added alongside `_and_if_to_nested` -- a smaller (3.6% of real
    records) but cheap-to-cover pattern from the same audit. Only a bare
    `if` test that's a plain expression (or a single `not`-wrapped
    expression) on one physical line -- `Compare`/`BoolOp` tests are
    excluded (an `if (a < b) == True:` corruption is a much less faithful
    "verbose real-world" example than the plain-name/call case)."""
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if test.lineno != test.end_lineno:
            continue
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            candidates.append((node, test.operand, False))
        elif not isinstance(test, (ast.Compare, ast.BoolOp)):
            candidates.append((node, test, True))
    if not candidates:
        return None

    node, operand, truthy = rng.choice(candidates)
    operand_text = ast.get_source_segment(code, operand)
    if operand_text is None:
        return None
    literal = "True" if truthy else "False"

    indent = " " * node.col_offset
    new_first_line = f"{indent}if {operand_text} == {literal}:"
    return _replace_source_lines(code, node.lineno, node.lineno, [new_first_line])


def make_simplify_bridge(
    code: str, tok, cfg, rng: Optional[random.Random] = None,
) -> Optional[Dict]:
    """Build one simplify bridging example from real (already-concise) code.

    Returns {type: "simplify", before, after} or None if unsuitable. `after`
    == the real concise code unchanged; `before` has one idiom mechanically
    expanded into a more verbose equivalent (listcomp->loop, or ternary
    return -> if/else) -- inverted direction from remove_dead_code/rename:
    the corruption here *adds* verbosity, the target removes it, matching
    the original bridging brief's simplify spec."""
    rng = rng or random
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    generators = [_listcomp_to_loop, _ternary_return_to_ifelse,
                  _augassign_to_expanded, _boolop_return_to_ifelse,
                  _and_if_to_nested, _plain_bool_to_redundant_compare]
    rng.shuffle(generators)
    before_text = None
    for gen in generators:
        before_text = gen(code, tree, rng)
        if before_text is not None:
            break
    if before_text is None or before_text == code:
        return None

    record = {"type": "simplify", "before": before_text, "after": code}
    if not _validate_text_record(record, tok, cfg):
        return None
    return record


def _collect_names(nodes, ctx_type) -> set:
    names = set()
    for n in nodes:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ctx_type):
                names.add(sub.id)
    return names


def _simple_function_candidates(tree: ast.AST) -> List[ast.FunctionDef]:
    """Top-level functions only (no class methods -- avoids self/cls
    complications), with a plain positional-only signature (no *args/**kwargs/
    defaults/kwonly) so call-forwarding is a trivial name-for-name copy.
    Used by deduplicate's earlier positional-only path; extract_method uses
    `_all_binding_names` instead since its helper can take an explicit
    parameter list regardless of the enclosing function's own signature
    shape."""
    return [
        n for n in getattr(tree, "body", [])
        if isinstance(n, ast.FunctionDef)
        and not n.args.vararg and not n.args.kwarg
        and not n.args.kwonlyargs and not n.args.defaults
        and n.body
    ]


def _all_binding_names(args: ast.arguments) -> List[str]:
    """Every name `args` binds inside the function, in a stable order:
    positional args, *vararg, keyword-only args, **kwarg. Used to decide
    which free variables in an extracted block are "available" without
    needing to reconstruct the original signature's exact shape."""
    names = [a.arg for a in getattr(args, "posonlyargs", [])] + [a.arg for a in args.args]
    if args.vararg:
        names.append(args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def make_extract_method_bridge(
    code: str, tok, cfg, rng: Optional[random.Random] = None,
) -> Optional[Dict]:
    """Build one extract_method bridging example from real code.

    Returns {type: "extract_method", before, after} or None if unsuitable.
    Picks a contiguous 1-2 statement block directly in a top-level function's
    own body (never nested inside a further if/for -- keeps indentation
    reuse safe, see below) whose free variable reads are a subset of that
    function's own parameter-like bindings (any signature shape) *or* names
    assigned by an earlier statement in the same function body -- passing the
    current value of an outer local through as a helper parameter is exactly
    what extraction does by hand, so this isn't a semantic stretch, just more
    coverage than "parameters only" (found via diagnostic testing: parameters-
    only was the dominant rejection cause, not the block-length/return
    restrictions). Blocks needing something genuinely unavailable at that
    point (not yet assigned, or from an enclosing scope) are still skipped.

    `after` == the real code, restructured: the block is replaced by a call
    to a new helper function (inserted before the enclosing function) whose
    body is the block's own source lines, byte-verbatim -- since the block is
    a *direct* child of the function body, its existing indentation is
    already exactly one level, which is already correct for a fresh top-level
    function's body, so no re-indentation is needed or attempted.
    `before` == the real code with the block inlined (i.e. the *pre-refactor*
    state) -- so direction here matches remove_dead_code/rename: `before`
    needs the edit, `after` is the already-refactored target."""
    rng = rng or random
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    candidates = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.body]
    rng.shuffle(candidates)

    for fn in candidates:
        param_names = _all_binding_names(fn.args)
        body = fn.body
        start_idx = 0
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            start_idx = 1                      # skip a leading docstring

        slots = list(range(start_idx, len(body)))
        rng.shuffle(slots)
        for i in slots:
            # names assigned by earlier statements in this same function --
            # a block may read these too, not just the function's own params
            earlier_assigned: List[str] = []
            seen = set()
            for s in body[:i]:
                for n in _collect_names([s], ast.Store):
                    if n not in seen:
                        seen.add(n)
                        earlier_assigned.append(n)
            name_order = param_names + [n for n in earlier_assigned if n not in param_names]
            available = set(name_order)

            for block_len in (2, 1):
                j = i + block_len
                if j > len(body):
                    continue
                stmts = body[i:j]
                # a return anywhere but the last statement changes control
                # flow once extracted -- skip rather than handle it
                if any(isinstance(s, ast.Return) for s in stmts[:-1]):
                    continue
                loads = _collect_names(stmts, ast.Load) - _collect_names(stmts, ast.Store)
                if not loads.issubset(available | _BUILTIN_NAMES):
                    continue

                used_names = [n for n in name_order if n in loads]
                result = _build_extract_method_record(code, fn, stmts, used_names)
                if result is not None and _validate_text_record(result, tok, cfg):
                    return result
    return None


def _build_extract_method_record(
    code: str, fn: ast.FunctionDef, stmts: List[ast.stmt], used_params: List[str],
) -> Optional[Dict]:
    existing_names = set(_collect_local_identifiers(ast.parse(code)))
    helper_name = None
    for suffix in ["_helper", "_extracted", "_inner"]:
        cand = f"{fn.name}{suffix}"
        if cand not in existing_names:
            helper_name = cand
            break
    if helper_name is None:
        return None

    lines = code.split("\n")
    first_ln, last_ln = stmts[0].lineno, stmts[-1].end_lineno
    if last_ln is None:
        return None
    block_lines = lines[first_ln - 1:last_ln]
    indent = " " * stmts[0].col_offset

    has_return = isinstance(stmts[-1], ast.Return) and stmts[-1].value is not None
    call_line = (f"{indent}return {helper_name}({', '.join(used_params)})" if has_return
                 else f"{indent}{helper_name}({', '.join(used_params)})")

    helper_def_lines = [f"def {helper_name}({', '.join(used_params)}):"] + block_lines + [""]

    # replace block first (strictly after fn's own def line -- doesn't shift
    # fn.lineno), then insert the helper before fn's def line
    new_lines = list(lines)
    new_lines[first_ln - 1:last_ln] = [call_line]
    new_lines[fn.lineno - 1:fn.lineno - 1] = helper_def_lines
    after_text = "\n".join(new_lines)

    # Bug fixed later: this used to assign the refactored (helper+call)
    # text to "before" and the raw original to "after" -- backwards, since
    # make_sft_sr treats "before" as the encoder input and diffs toward
    # "after" as the decoder target. That trained the model to *undo*
    # extraction (inline the call, delete the helper) instead of performing
    # it. before = raw/original (needs the edit), after = refactored (goal),
    # matching every other generator in this module and the real SFT data.
    if after_text == code:
        return None
    return {"type": "extract_method", "before": code, "after": after_text}


def _find_singleton_literal(fn: ast.FunctionDef) -> List[ast.Constant]:
    """str/int/float literals (not bool, not empty string) inside `fn`'s
    body -- skipping the docstring and anything inside a nested def/lambda/
    class (an inner scope's own literal isn't safe to hoist into the outer
    function's new parameter) -- each confined to a single physical source
    line (true for any non-triple-quoted literal), so its span can be
    substituted unambiguously via line/column offsets alone."""
    body = fn.body
    start_idx = 1 if (
        body and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ) else 0

    nested_scopes: set = set()
    for stmt in body[start_idx:]:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                nested_scopes.update(ast.walk(node))

    consts = []
    for stmt in body[start_idx:]:
        for node in ast.walk(stmt):
            if node in nested_scopes:
                continue
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, (str, int, float))
                    and not isinstance(node.value, bool)
                    and node.value != ""
                    and node.lineno == node.end_lineno
                    and node.lineno != fn.lineno):
                consts.append(node)
    return consts


def _variant_literal_value(value):
    """A plausible, type-matched 'different' literal for the second wrapper
    -- deterministic and never fabricated out of nowhere, mirroring the
    module's `_DEGRADED_POOL` philosophy of small, mechanical substitutions."""
    if isinstance(value, str):
        return f"{value}_2"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    return None


def _has_top_level_return(fn: ast.FunctionDef) -> bool:
    """Whether `fn` has a value-returning `return` directly in its own body
    -- not inside a nested def/lambda/class, which has its own return."""
    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if walk(child):
                return True
        return False
    return walk(fn)


def _build_shared_helper_record(
    code: str, fn: ast.FunctionDef, const_node: ast.Constant,
) -> Optional[Dict]:
    """Build one {before, after} pair for the shared-helper-extraction shape:
    `before` has `fn` (unchanged) plus a second near-duplicate copy of it
    (fresh name, one literal swapped for a type-matched variant) -- the
    pre-refactor, duplicated state. `after` replaces both with (1) a new
    function -- `fn`'s own body, byte-identical except the literal is now a
    parameter -- and (2) two thin one-line wrappers (original name +
    variant name) each calling it with their own literal, matching the
    dominant shape in the real generated SFT reference (e.g. `hash1 =
    hashlib.sha256(user_id...)` / `hash2 = hashlib.sha256(admin_id...)` ->
    `hash_user_id(x)` helper called twice; or two near-duplicate functions
    like `migrate_table1`/`migrate_table2` -> one parameterized
    `migrate_table(name)` plus two wrappers)."""
    if fn.decorator_list or fn.col_offset != 0:
        return None
    old_header = f"def {fn.name}("
    param_names = _all_binding_names(fn.args)

    existing_names = set(_collect_local_identifiers(ast.parse(code))) | {fn.name}
    param_name = next((c for c in ("value", "arg", "extra", "detail", "data")
                        if c not in existing_names and c not in param_names), None)
    wrapper2_name = next((f"{fn.name}{s}" for s in ("_2", "_b", "_alt")
                          if f"{fn.name}{s}" not in existing_names), None)
    shared_name = next((f"{fn.name}{s}" for s in ("_impl", "_shared", "_core")
                        if f"{fn.name}{s}" not in existing_names and f"{fn.name}{s}" != wrapper2_name), None)
    if param_name is None or wrapper2_name is None or shared_name is None:
        return None

    variant_value = _variant_literal_value(const_node.value)
    if variant_value is None:
        return None

    lines = code.split("\n")
    orig_fn_lines = lines[fn.lineno - 1: fn.end_lineno]
    if old_header not in orig_fn_lines[0]:
        return None
    rel_line = const_node.lineno - fn.lineno

    # shared helper: fn's own body, def-line renamed + parameterized, the
    # literal's own line spliced from param_name in place of the literal
    shared_lines = list(orig_fn_lines)
    new_params = ", ".join(param_names + [param_name])
    shared_lines[0] = shared_lines[0].replace(old_header, f"def {shared_name}(", 1).replace(
        f"({', '.join(param_names)})", f"({new_params})", 1)
    line = shared_lines[rel_line]
    shared_lines[rel_line] = line[:const_node.col_offset] + param_name + line[const_node.end_col_offset:]

    # duplicate clone for "before": fn's own body, def-line renamed only,
    # literal's own line spliced with the variant value
    clone_lines = list(orig_fn_lines)
    clone_lines[0] = clone_lines[0].replace(old_header, f"def {wrapper2_name}(", 1)
    cline = clone_lines[rel_line]
    clone_lines[rel_line] = cline[:const_node.col_offset] + repr(variant_value) + cline[const_node.end_col_offset:]

    has_return = _has_top_level_return(fn)

    def wrapper_lines(name, literal_value):
        call_expr = f"{shared_name}({', '.join(param_names + [repr(literal_value)])})"
        call_line = f"    return {call_expr}" if has_return else f"    {call_expr}"
        return [f"def {name}({', '.join(param_names)}):", call_line]

    wrapper_a_lines = wrapper_lines(fn.name, const_node.value)
    wrapper_b_lines = wrapper_lines(wrapper2_name, variant_value)

    before_fn = lines[:fn.lineno - 1]
    after_fn = lines[fn.end_lineno:]

    before_text = "\n".join(before_fn + orig_fn_lines + [""] + clone_lines + [""] + after_fn)
    after_text = "\n".join(before_fn + shared_lines + [""] + wrapper_a_lines + [""] + wrapper_b_lines + [""] + after_fn)
    if before_text == after_text:
        return None
    return {"type": "deduplicate", "before": before_text, "after": after_text}


_MAX_SHARED_HELPER_FN_TRIES = 5
_MAX_SHARED_HELPER_CONST_TRIES = 3


def _synthesize_shared_helper_record(
    code: str, tree: ast.AST, rng: random.Random, tok, cfg,
) -> Optional[Dict]:
    """Bounded search: each `_validate_text_record` call re-tokenizes and
    round-trips both `before`/`after` through `apply_sr`, which is expensive
    enough that trying every literal in every candidate function of every
    snippet (some source files have dozens of each) made a full-corpus scan
    take hours instead of minutes. Capped to the first few candidates/
    literals after shuffling -- still a uniform-random sample of what a
    snippet offers, just not an exhaustive one."""
    candidates = [n for n in _simple_function_candidates(tree) if not n.decorator_list]
    rng.shuffle(candidates)
    for fn in candidates[:_MAX_SHARED_HELPER_FN_TRIES]:
        consts = _find_singleton_literal(fn)
        if not consts:
            continue
        rng.shuffle(consts)
        for const_node in consts[:_MAX_SHARED_HELPER_CONST_TRIES]:
            record = _build_shared_helper_record(code, fn, const_node)
            if record is not None and _validate_text_record(record, tok, cfg):
                return record
    return None


def make_deduplicate_bridge(
    code: str, tok, cfg, rng: Optional[random.Random] = None,
) -> Optional[Dict]:
    """Build one deduplicate bridging example from real code.

    Rewritten later: the original version synthesized a byte-identical
    duplicate *function* collapsed into a `*args/**kwargs` forwarder. Direct
    corpus check against the real SFT deduplicate data (3000 records) found
    that shape in **0** of them -- every real example instead extracts a
    shared helper from a near-duplicate *statement* that differs only in
    which single variable it reads (e.g. `hash1 = hashlib.sha256(user_id...)`
    / `hash2 = hashlib.sha256(admin_id...)` -> one `hash_user_id(x)` helper
    called twice). This version synthesizes that shape instead: picks a
    single-physical-line `target = expr` assignment inside a top-level
    function whose free-variable reads include at least one name already
    available at that point (param or earlier-assigned local, same
    eligibility rule as make_extract_method_bridge), duplicates the
    statement with exactly one of those reads substituted for a different
    available name (and the assignment target renamed to avoid clobbering)
    to build `before` (now has two near-duplicate statements, pre-refactor),
    then extracts a helper parameterized on the varying name and rewrites
    both statements as calls to build `after` (the refactored, goal state).
    Name substitution operates on AST-located character spans of the
    matching `Load` `Name` nodes (not regex), so it can't accidentally hit
    an identifier-shaped substring inside a string or comment.

    Corrected later: a "whole duplicate function, delegates
    verbatim to its sibling" generator was added and briefly made primary,
    audited (wrongly) against `data/supervised_finetuning.jsonl` (the small
    curated set) where it found 0 matches for its own shape -- so that
    generator was reverted outright. The *actual* reference this project
    trains SFT against is the larger `data/finetune_data/
    supervised_finetuning_generated.jsonl` (75,000 records, 15,000/task).
    Checking deduplicate there directly (via before/after top-level `def`
    counts) shows a clear dominant shape covering ~85% of its 15,000
    records: N near-duplicate call sites/functions differing by exactly one
    literal argument collapse to N+1 defs -- one NEW shared function
    parameterized on that literal, plus each site rewritten as a thin call
    (e.g. `migrate_table1`/`migrate_table2`, each with a hardcoded table
    name, both replaced by calls into one new `migrate_table(name)`). That
    is what `_synthesize_shared_helper_record` above builds, and it is now
    tried first. The single-line pattern below (identical to the untouched
    original version) still matches a real, if smaller, slice of the
    reference data and is kept as the fallback when no function in the
    snippet has a synthesizable literal."""
    rng = rng or random
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    record = _synthesize_shared_helper_record(code, tree, rng, tok, cfg)
    if record is not None:
        return record

    candidates = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.body]
    rng.shuffle(candidates)

    for fn in candidates:
        param_names = _all_binding_names(fn.args)
        body = fn.body
        start_idx = 1 if (
            body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ) else 0

        slots = list(range(start_idx, len(body)))
        rng.shuffle(slots)
        for i in slots:
            stmt = body[i]
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue
            if stmt.lineno != stmt.end_lineno:
                continue                        # single physical line only

            earlier_assigned: List[str] = []
            seen = set()
            for s in body[:i]:
                for n in _collect_names([s], ast.Store):
                    if n not in seen:
                        seen.add(n)
                        earlier_assigned.append(n)
            available = set(param_names) | seen

            load_names = _collect_names([stmt.value], ast.Load)
            varying_candidates = sorted(load_names & available)
            if not varying_candidates:
                continue
            rng.shuffle(varying_candidates)

            record = _build_deduplicate_record(code, fn, stmt, varying_candidates,
                                                available - load_names, rng)
            if record is not None and _validate_text_record(record, tok, cfg):
                return record
    return None


def _build_deduplicate_record(
    code: str, fn: ast.FunctionDef, stmt: ast.Assign,
    varying_candidates: List[str], other_available: set, rng: random.Random,
) -> Optional[Dict]:
    for varying in varying_candidates:
        second_candidates = sorted(other_available - {varying})
        if not second_candidates:
            continue
        second_name = rng.choice(second_candidates)

        value_text = ast.get_source_segment(code, stmt.value)
        if value_text is None:
            continue
        value_start_col = stmt.value.col_offset
        value_start_line = stmt.value.lineno

        # locate this varying name's Load spans within stmt.value, as
        # (col_offset, end_col_offset) relative to value_text -- restricted
        # to stmt.value's own single source line (guaranteed by the
        # lineno==end_lineno check on stmt itself in the caller)
        spans = [
            (n.col_offset - value_start_col, n.end_col_offset - value_start_col)
            for n in ast.walk(stmt.value)
            if isinstance(n, ast.Name) and n.id == varying
            and isinstance(n.ctx, ast.Load) and n.lineno == value_start_line
        ]
        if not spans:
            continue
        spans.sort(reverse=True)
        dup_value_text = value_text
        for s, e in spans:
            dup_value_text = dup_value_text[:s] + second_name + dup_value_text[e:]

        orig_target = stmt.targets[0].id
        dup_target = f"{orig_target}2"
        existing_names = set(_collect_local_identifiers(ast.parse(code))) | {fn.name}
        if dup_target in existing_names:
            continue

        indent = " " * stmt.col_offset
        dup_line = f"{indent}{dup_target} = {dup_value_text}"

        lines = code.split("\n")
        insert_at = stmt.end_lineno            # 1-indexed end line -> insert right after it
        before_lines = list(lines)
        before_lines[insert_at:insert_at] = [dup_line]
        before_text = "\n".join(before_lines)
        try:
            ast.parse(before_text)
        except SyntaxError:
            continue

        helper_name = None
        for base in [f"compute_{orig_target}", f"get_{orig_target}", f"{orig_target}_from_value"]:
            if base not in existing_names:
                helper_name = base
                break
        if helper_name is None:
            continue

        helper_def_lines = [f"def {helper_name}({varying}):", f"    return {value_text}", ""]
        call_line_orig = f"{indent}{orig_target} = {helper_name}({varying})"
        call_line_dup = f"{indent}{dup_target} = {helper_name}({second_name})"

        after_lines = list(lines)
        after_lines[stmt.lineno - 1:stmt.end_lineno] = [call_line_orig, call_line_dup]
        after_lines[fn.lineno - 1:fn.lineno - 1] = helper_def_lines
        after_text = "\n".join(after_lines)
        try:
            ast.parse(after_text)
        except SyntaxError:
            continue

        if before_text == after_text:
            continue
        return {"type": "deduplicate", "before": before_text, "after": after_text}
    return None
