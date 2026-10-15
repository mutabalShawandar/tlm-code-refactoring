# -- SFT: before/after -> SEARCH/REPLACE target (line-level minimal hunks) ------
#
# Mirrors the pretrain decoder format learned in make_sr_corruption:
#   enc: <bos> <task_token> [before code ids] <eos>
#   dec: <bos> (<search> old_block <replace> new_block)* <eos>
#
# Unlike pretrain (span corruption), SFT targets are derived from a real
# before->after edit via a line-level diff. All work is done in BPE-id space
# (not text), because the tokenizer is stateful (indent/dedent, normalize_code)
# and text lines cannot be tokenized independently and concatenated.

from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

# type string -> encoder task special token  (project spec: 9..13)
TASK_TOKEN_NAME = {
    "rename":           "rename",
    "simplify":         "simplify",
    "deduplicate":      "deduplicate",
    "remove_dead_code": "remove_dead_code",
    "extract_method":   "extract_method",
}

# type string -> PLE task_id  (project spec: 1..5; 0 reserved for pretrain)
TASK_ID = {
    "rename":           1,
    "simplify":         2,
    "deduplicate":      3,
    "remove_dead_code": 4,
    "extract_method":   5,
}


def split_lines_by_newline(ids: List[int], newline_id: int) -> List[List[int]]:
    """Split an id stream into per-line id lists, dropping the <newline> delimiters.
    A trailing <newline> (always emitted by normalize_code) yields no trailing
    empty line; an explicit blank line yields an empty list element."""
    lines: List[List[int]] = []
    cur: List[int] = []
    for tid in ids:
        if tid == newline_id:
            lines.append(cur)
            cur = []
        else:
            cur.append(tid)
    if cur:                      # content after the last newline (rare)
        lines.append(cur)
    return lines


def join_lines(lines: List[List[int]], newline_id: int) -> List[int]:
    """Inverse of split_lines_by_newline (newline between lines, no trailing one)."""
    out: List[int] = []
    for i, ln in enumerate(lines):
        if i > 0:
            out.append(newline_id)
        out.extend(ln)
    return out


def diff_hunks(
    before_lines: List[List[int]],
    after_lines: List[List[int]],
) -> List[Tuple[List[List[int]], List[List[int]]]]:
    """Line-level minimal hunks. Returns list of (search_lines, replace_lines).
    A SEARCH block that is empty (pure insertion) or that consists of exactly
    one blank line (a lone-blank-line delete/replace) is anchored to an
    adjacent equal line so every SEARCH block is non-empty and locatable when
    applied to `before`.

    Both cases exist because `join_lines`/`split_lines_by_newline` cannot
    distinguish "one blank line" from "zero lines" (both serialize to zero
    tokens between the <search>/<replace> markers), so either one round-trips
    back as an empty SEARCH block and `apply_sr` rejects it as unlocatable.
    Insertions next to a blank separator (the common case: inserting a new
    helper next to the blank line between two functions) and deletions of a
    lone blank line are exactly where this bites. Fix: extend the anchor
    across consecutive blank lines until it includes a non-blank line.

    The same collapse can also happen at the *trailing* end of a multi-line
    SEARCH or REPLACE block: `join_lines` places separators only *between*
    lines (N-1 for N lines), so a blank line landing last is indistinguishable
    from "no more lines" on the way back. If that happens, the block silently
    loses its last line -- in the SEARCH case this can make the block match
    content already consumed by an earlier hunk instead of the intended
    position. Fix: extend both SEARCH and REPLACE forward by the same
    (verified-identical) next line from the surrounding equal context until
    neither ends in a blank line.

    Two further, distinct problems (found via round-trip testing on real
    deduplicate/extract_method data):

    1. **Duplicate content elsewhere in the document.** A minimal SEARCH block
       (e.g. one function body shared verbatim by several near-duplicate
       functions, or a common one-line anchor like a bare `else:` that recurs
       at several nesting levels) can match more than one position in
       `before`. `apply_sr`'s forward-cursor scan then finds whichever
       occurrence comes first, not necessarily the intended one. Fix: extend
       SEARCH/REPLACE outward with adjacent equal context (same mechanism as
       the anchor fixes above) until the block is a *globally unique*
       contiguous subsequence of `before` -- the same principle unified/context
       diffs use to make a hunk's location unambiguous.
    2. **Two hunks independently claiming the same bordering anchor line.**
       When two separate edits (e.g. two insertions) sit on either side of a
       single shared equal line, each one's anchor-extension logic (above)
       independently borrows that same line as part of its own SEARCH. The
       first hunk's match consumes it at apply-time, so the second hunk's
       identical SEARCH request can no longer be found forward of the cursor
       -- even though the line's content was globally unique in `before`. Fix:
       after building all hunks, merge any two consecutive hunks whose
       claimed `before`-line ranges overlap into a single hunk, so the shared
       line is claimed once."""
    a = [tuple(x) for x in before_lines]
    b = [tuple(x) for x in after_lines]
    sm = SequenceMatcher(a=a, b=b, autojunk=False)

    def _lossy(lines: List[List[int]]) -> bool:
        return not lines or (len(lines) == 1 and not lines[0])

    def _is_unique(search: List[List[int]]) -> bool:
        s = tuple(tuple(x) for x in search)
        n = len(s)
        if n == 0:
            return True
        count = 0
        for start in range(len(a) - n + 1):
            if tuple(a[start:start + n]) == s:
                count += 1
                if count > 1:
                    return False
        return count == 1

    raw: List[Tuple[int, int, List[List[int]], List[List[int]]]] = []
    prev_end = 0                             # rightmost `before`-index already claimed
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        search = [list(x) for x in a[i1:i2]]
        replace = [list(x) for x in b[j1:j2]]
        if _lossy(search) and i1 == i2 and i1 == prev_end and raw:
            # Pure insertion touching the previous hunk with zero gap: fold it
            # directly into that hunk's REPLACE tail instead of anchoring at
            # all. No `before` content is consumed by an insertion (i1==i2), so
            # there is nothing to search for -- anchoring here would need to
            # borrow context from one side or the other, but with zero gap
            # there may be no non-blank line available on either side without
            # invading a neighboring hunk's own required content (found via
            # round-trip testing -- the previous backward/forward anchor logic
            # could steal a line the *next* hunk's unmodified SEARCH needed).
            # Folding avoids the anchor question entirely: the previous hunk's
            # SEARCH already matches up to exactly this point.
            p_i1, p_i2, p_search, p_replace = raw[-1]
            raw[-1] = (p_i1, p_i2, p_search, p_replace + replace)
            prev_end = p_i2
            continue
        if _lossy(search):                   # anchor on adjacent equal line(s)
            anchored = False
            if i1 > 0:
                # Never extend backward into a previous hunk's already-claimed
                # territory: that territory's REPLACE content there is the real
                # transformed output, not raw `a` content, so borrowing it as a
                # plain anchor (as if unchanged) would silently corrupt it.
                start = i1 - 1
                while start > prev_end and all(not a[k] for k in range(start, i1)):
                    start -= 1
                if start >= prev_end and any(a[k] for k in range(start, i1)):
                    anchor = [list(a[k]) for k in range(start, i1)]
                    search = anchor + search
                    replace = anchor + replace
                    j1 -= (i1 - start)
                    i1 = start
                    anchored = True
            if not anchored and i2 < len(a):   # fall back to anchoring forward
                end = i2 + 1
                while end < len(a) and all(not a[k] for k in range(i2, end)):
                    end += 1
                anchor = [list(a[k]) for k in range(i2, end)]
                search = search + anchor
                replace = replace + anchor
                j2 += (end - i2)
                i2 = end
            # else: no valid anchor available; leave as-is (degenerate, filtered later)

        while ((search and not search[-1]) or (replace and not replace[-1])) \
                and i2 < len(a) and j2 < len(b) and a[i2] == b[j2]:
            search.append(list(a[i2]))
            replace.append(list(b[j2]))
            i2 += 1
            j2 += 1

        while not _is_unique(search):        # disambiguate vs. duplicate content
            if i1 > prev_end and j1 > 0 and a[i1 - 1] == b[j1 - 1]:
                line = list(a[i1 - 1])
                search = [line] + search
                replace = [line] + replace
                i1 -= 1
                j1 -= 1
            elif i2 < len(a) and j2 < len(b) and a[i2] == b[j2]:
                line = list(a[i2])
                search = search + [line]
                replace = replace + [line]
                i2 += 1
                j2 += 1
            else:
                break                        # no more context available; leave ambiguous

        raw.append((i1, i2, search, replace))
        prev_end = i2

    merged: List[Tuple[int, int, List[List[int]], List[List[int]]]] = []
    for i1, i2, search, replace in raw:
        if merged:
            p_i1, p_i2, p_search, p_replace = merged[-1]
            k = p_i2 - i1                    # overlap in `before`-line count
            if 0 < k <= len(p_search) and k <= len(search) \
                    and k <= len(p_replace) and k <= len(replace) \
                    and p_search[-k:] == search[:k] \
                    and p_replace[-k:] == replace[:k]:
                # Only safe to merge if the overlapping content is verified
                # identical on both sides -- true when both hunks reached it via
                # plain equal-context anchoring, false if either side's overlap
                # is real transformed content (see prev_end guard above; this is
                # a belt-and-suspenders check, not the primary defense).
                merged[-1] = (
                    p_i1, max(p_i2, i2),
                    p_search + search[k:],
                    p_replace[:len(p_replace) - k] + replace,
                )
                continue
        merged.append((i1, i2, search, replace))

    return [(search, replace) for _, _, search, replace in merged]


def make_sft_sr(
    record: Dict, tok, cfg, max_src_len: int = 512, max_tgt_len: int = 512,
) -> Optional[Tuple[List[int], List[int], int]]:
    """Build one SFT sample from a {before, after, type} record.

    Returns (enc_ids, dec_ids, task_id) or None if there is no edit, the task
    type is unknown, or either sequence exceeds its length budget.
      enc_ids: <bos> <task_token> [before ids] <eos>
      dec_ids: <bos> (<search> old <replace> new)* <eos>
    """
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

    hunks = diff_hunks(before_lines, after_lines)
    if not hunks:                            # identical before/after -> no signal
        return None

    enc_ids = [BOS, TASK_TOK] + before_ids + [EOS]

    dec_body: List[int] = []
    for search_lines, replace_lines in hunks:
        dec_body += [SEARCH] + join_lines(search_lines, NL)
        dec_body += [REPLACE] + join_lines(replace_lines, NL)
    dec_ids = [BOS] + dec_body + [EOS]

    if len(enc_ids) > max_src_len or len(dec_ids) > max_tgt_len:
        return None

    return enc_ids, dec_ids, TASK_ID[ttype]


def make_full_after(
    record: Dict, tok, cfg, max_src_len: int = 512, max_tgt_len: int = 512,
) -> Optional[Tuple[List[int], List[int], int]]:
    """Build one full-after SFT sample from a {before, after, type} record.

    Returns (enc_ids, dec_ids, task_id) or None if the task type is unknown,
    before == after, or either sequence exceeds its length budget.
      enc_ids: <bos> <task_token> [before ids] <eos>
      dec_ids: <bos> [after ids] <eos>
    """
    ttype = record.get("type")
    if ttype not in TASK_ID:
        return None

    BOS = cfg.special_token_ids["bos"]
    EOS = cfg.special_token_ids["eos"]
    TASK_TOK = cfg.special_token_ids[TASK_TOKEN_NAME[ttype]]

    before_ids = tok.encode(record["before"])
    after_ids  = tok.encode(record["after"])
    if not before_ids or before_ids == after_ids:
        return None

    enc_ids = [BOS, TASK_TOK] + before_ids + [EOS]
    dec_ids = [BOS] + after_ids + [EOS]

    if len(enc_ids) > max_src_len or len(dec_ids) > max_tgt_len:
        return None

    return enc_ids, dec_ids, TASK_ID[ttype]


def apply_sr(
    before_ids: List[int], dec_ids: List[int], cfg,
) -> Optional[List[int]]:
    """Apply a decoder SEARCH/REPLACE sequence to before_ids -> predicted after_ids.
    Used by the applied_ok eval. Hunks are matched as contiguous line subsequences
    in document order. Returns None if any SEARCH block cannot be located.

    Each REPLACE block's own leading <indent>/<dedent> tokens are trusted as-is
    (no re-anchoring to SEARCH's prefix). An earlier version re-anchored the first
    REPLACE line's indent/dedent run onto the matched SEARCH block's leading tokens,
    on the assumption that none of the 5 SFT tasks change nesting depth. That
    assumption is false -- remove_dead_code, extract_method, and deduplicate all
    routinely shift what follows to a different depth -- and SEARCH/REPLACE blocks
    can have different line counts, so SEARCH's first line isn't even the
    structurally corresponding line to REPLACE's first line. The reanchor corrupted
    gold round-trip reconstruction on ~15-43% of those task types. REPLACE's own
    declared prefix is always correct for gold data (derived from real `after`
    tokenization) and is exactly what the model is trained to predict via
    cross-attention over `before`."""
    BOS = cfg.special_token_ids["bos"]
    EOS = cfg.special_token_ids["eos"]
    NL = cfg.special_token_ids["newline"]
    SEARCH = cfg.special_token_ids["search"]
    REPLACE = cfg.special_token_ids["replace"]

    # parse dec into (search_lines, replace_lines) pairs
    body = [t for t in dec_ids if t not in (BOS, EOS)]
    pairs: List[Tuple[List[List[int]], List[List[int]]]] = []
    i = 0
    while i < len(body):
        if body[i] != SEARCH:
            return None
        i += 1
        s_start = i
        while i < len(body) and body[i] != REPLACE:
            i += 1
        if i >= len(body):
            return None
        search_block = body[s_start:i]
        i += 1                                # skip REPLACE
        r_start = i
        while i < len(body) and body[i] != SEARCH:
            i += 1
        replace_block = body[r_start:i]
        pairs.append((
            split_lines_by_newline(search_block, NL),
            split_lines_by_newline(replace_block, NL),
        ))

    lines = split_lines_by_newline(before_ids, NL)
    cursor = 0
    out: List[List[int]] = []
    for search_lines, replace_lines in pairs:
        a = [tuple(x) for x in lines]
        s = [tuple(x) for x in search_lines]
        if not s:
            return None
        # find s as a contiguous subsequence at/after cursor
        pos = -1
        for start in range(cursor, len(a) - len(s) + 1):
            if a[start:start + len(s)] == s:
                pos = start
                break
        if pos < 0:
            return None
        out.extend(lines[cursor:pos])
        out.extend(replace_lines)
        cursor = pos + len(s)
    out.extend(lines[cursor:])

    return join_lines(out, NL)
