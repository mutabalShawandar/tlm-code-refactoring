# -- SFT/bridging SEARCH/REPLACE construction, text-line variant --------------
#
# Everything in `src/codet5/` is HF-T5-specific (official CodeT5-small weights,
# real RobertaTokenizer/T5Config) and deliberately kept separate from the rest
# of `src/` (TLM's own architecture, tokenizer, and training pipeline) -- no
# imports either direction. `src/data_set/sft_corruption.py` remains the
# canonical implementation for anything TLM-specific; this module exists only
# because HF's real tokenizer/vocab makes that one inapplicable here.
#
# Same construction as sft_corruption.py's diff_hunks/make_sft_sr/apply_sr, but
# operating on plain text lines (List[str]) instead of TLM-tokenizer BPE-id
# lines (List[List[int]]) -- for use with a real HF tokenizer (e.g. CodeT5-
# small's RobertaTokenizer) whose vocab/encoding this project doesn't control.
# Reusing the pretrained model's own tokenizer/weights untouched (only adding
# a handful of new control tokens) means SEARCH/REPLACE line-diffing has to
# happen at the text level, since token-id lines would depend on that
# tokenizer's own (unknown, uncontrolled) segmentation of whitespace/newlines.
#
# `diff_hunks_text` is a line-for-line port of `diff_hunks` in sft_corruption.py
# -- same algorithm (SequenceMatcher opcodes, blank-line anchoring, duplicate-
# content disambiguation, adjacent-hunk merging), same edge cases, just with
# the atomic "line" unit swapped from an id-list to a string. Keep the two in
# sync if either's diff logic changes -- deliberately not shared via a common
# generic implementation, to avoid risking either's already-tested behavior.

from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

TASK_TOKEN_NAME = {
    "rename":           "rename",
    "simplify":         "simplify",
    "deduplicate":      "deduplicate",
    "remove_dead_code": "remove_dead_code",
    "extract_method":   "extract_method",
}

TASK_ID = {
    "rename":           1,
    "simplify":         2,
    "deduplicate":      3,
    "remove_dead_code": 4,
    "extract_method":   5,
}


def split_lines(text: str) -> List[str]:
    """Text equivalent of split_lines_by_newline -- NOT plain `str.split`.
    `"".split("\\n")` returns `['']` (one blank line), but the id-space
    original returns `[]` (zero lines) for empty input (its loop never
    appends an empty trailing `cur`). Diverging from that breaks the
    "empty search/replace block == zero lines" convention `diff_hunks_text`'s
    `_lossy` check and `apply_sr_text` both rely on -- a genuine deletion
    (empty REPLACE content) would otherwise round-trip back with a spurious
    extra blank line. Special-cased here so `split_lines("") == []`, matching
    `split_lines_by_newline`'s behavior on an empty id list exactly."""
    return text.split("\n") if text != "" else []


def join_lines(lines: List[str]) -> str:
    """Inverse of split_lines."""
    return "\n".join(lines)


def diff_hunks_text(
    before_lines: List[str],
    after_lines: List[str],
) -> List[Tuple[List[str], List[str]]]:
    """Line-level minimal hunks over text lines. Returns list of
    (search_lines, replace_lines). Direct text-line port of `diff_hunks` in
    sft_corruption.py -- see that function's docstring for the full
    rationale behind blank-line anchoring, duplicate-content disambiguation,
    and adjacent-hunk merging; the logic here is unchanged, only the atomic
    line representation (str instead of List[int]) differs."""
    a = list(before_lines)
    b = list(after_lines)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)

    def _lossy(lines: List[str]) -> bool:
        return not lines or (len(lines) == 1 and lines[0] == "")

    def _is_unique(search: List[str]) -> bool:
        n = len(search)
        if n == 0:
            return True
        count = 0
        for start in range(len(a) - n + 1):
            if a[start:start + n] == search:
                count += 1
                if count > 1:
                    return False
        return count == 1

    raw: List[Tuple[int, int, List[str], List[str]]] = []
    prev_end = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        search = list(a[i1:i2])
        replace = list(b[j1:j2])
        if _lossy(search) and i1 == i2 and i1 == prev_end and raw:
            p_i1, p_i2, p_search, p_replace = raw[-1]
            raw[-1] = (p_i1, p_i2, p_search, p_replace + replace)
            prev_end = p_i2
            continue
        if _lossy(search):
            anchored = False
            if i1 > 0:
                start = i1 - 1
                while start > prev_end and all(a[k] == "" for k in range(start, i1)):
                    start -= 1
                if start >= prev_end and any(a[k] != "" for k in range(start, i1)):
                    anchor = list(a[start:i1])
                    search = anchor + search
                    replace = anchor + replace
                    j1 -= (i1 - start)
                    i1 = start
                    anchored = True
            if not anchored and i2 < len(a):
                end = i2 + 1
                while end < len(a) and all(a[k] == "" for k in range(i2, end)):
                    end += 1
                anchor = list(a[i2:end])
                search = search + anchor
                replace = replace + anchor
                j2 += (end - i2)
                i2 = end

        while ((len(search) > 0 and search[-1] == "") or (len(replace) > 0 and replace[-1] == "")) \
                and i2 < len(a) and j2 < len(b) and a[i2] == b[j2]:
            search.append(a[i2])
            replace.append(b[j2])
            i2 += 1
            j2 += 1

        while not _is_unique(search):
            if i1 > prev_end and j1 > 0 and a[i1 - 1] == b[j1 - 1]:
                line = a[i1 - 1]
                search = [line] + search
                replace = [line] + replace
                i1 -= 1
                j1 -= 1
            elif i2 < len(a) and j2 < len(b) and a[i2] == b[j2]:
                line = a[i2]
                search = search + [line]
                replace = replace + [line]
                i2 += 1
                j2 += 1
            else:
                break

        raw.append((i1, i2, search, replace))
        prev_end = i2

    merged: List[Tuple[int, int, List[str], List[str]]] = []
    for i1, i2, search, replace in raw:
        if merged:
            p_i1, p_i2, p_search, p_replace = merged[-1]
            k = p_i2 - i1
            if 0 < k <= len(p_search) and k <= len(search) \
                    and k <= len(p_replace) and k <= len(replace) \
                    and p_search[-k:] == search[:k] \
                    and p_replace[-k:] == replace[:k]:
                merged[-1] = (
                    p_i1, max(p_i2, i2),
                    p_search + search[k:],
                    p_replace[:len(p_replace) - k] + replace,
                )
                continue
        merged.append((i1, i2, search, replace))

    return [(search, replace) for _, _, search, replace in merged]


def apply_sr_text(before_text: str, hunks: List[Tuple[List[str], List[str]]]) -> Optional[str]:
    """Apply parsed (search_lines, replace_lines) hunks to `before_text` ->
    predicted after text. Text-line port of `apply_sr` in sft_corruption.py.
    Returns None if any SEARCH block cannot be located as a contiguous
    subsequence of `before_text`'s lines at/after the running cursor."""
    lines = split_lines(before_text)
    cursor = 0
    out: List[str] = []
    for search_lines, replace_lines in hunks:
        if not search_lines:
            return None
        pos = -1
        for start in range(cursor, len(lines) - len(search_lines) + 1):
            if lines[start:start + len(search_lines)] == search_lines:
                pos = start
                break
        if pos < 0:
            return None
        out.extend(lines[cursor:pos])
        out.extend(replace_lines)
        cursor = pos + len(search_lines)
    out.extend(lines[cursor:])
    return join_lines(out)


def parse_sr_pairs_text(dec_text: str, search_marker: str, replace_marker: str) -> List[Tuple[List[str], List[str]]]:
    """Parse a decoded SEARCH/REPLACE string (markers as literal substrings,
    e.g. "<search>"/"<replace>") into (search_lines, replace_lines) pairs.
    Mirrors sft_corruption.py's token-id `apply_sr` parse loop, but splitting
    on literal marker text instead of special token ids -- used both to
    build training targets (make_sft_sr_text below) and to parse a model's
    decoded prediction at eval time."""
    pairs: List[Tuple[List[str], List[str]]] = []
    body = dec_text
    while True:
        s_idx = body.find(search_marker)
        if s_idx < 0:
            break
        body = body[s_idx + len(search_marker):]
        r_idx = body.find(replace_marker)
        if r_idx < 0:
            return pairs if pairs else []
        search_block = body[:r_idx]
        body = body[r_idx + len(replace_marker):]
        next_s = body.find(search_marker)
        if next_s < 0:
            replace_block = body
            body = ""
        else:
            replace_block = body[:next_s]
        pairs.append((split_lines(search_block), split_lines(replace_block)))
        if next_s < 0:
            break
    return pairs


def build_sr_target_text(
    hunks: List[Tuple[List[str], List[str]]], search_marker: str, replace_marker: str,
) -> str:
    """Inverse of parse_sr_pairs_text: serialize hunks into the decoder
    target string, e.g. "<search>old<replace>new<search>old2<replace>new2"."""
    out = []
    for search_lines, replace_lines in hunks:
        out.append(search_marker + join_lines(search_lines))
        out.append(replace_marker + join_lines(replace_lines))
    return "".join(out)


def make_sft_sr_text(
    record: Dict, task_marker_fmt: str, search_marker: str, replace_marker: str,
    max_src_chars: int = 4000, max_tgt_chars: int = 4000,
) -> Optional[Tuple[str, str, int]]:
    """Build one SFT sample from a {before, after, type} record, text-level.

    Returns (enc_text, dec_text, task_id) or None if there is no edit, the
    task type is unknown, or either text exceeds its character budget (a
    cheap proxy check before tokenization -- the real token-length check
    happens after tokenizing, same as the id-space version's max_src_len/
    max_tgt_len, just deferred to the caller since tokenization isn't done
    here).
      enc_text: "{task_marker}{before}"
      dec_text: "({search_marker}old{replace_marker}new)*"
    `task_marker_fmt` is e.g. "<{}>" -- formatted with the task name (project's
    task-token convention, just as literal text here instead of a TLM special
    token id)."""
    ttype = record.get("type")
    if ttype not in TASK_ID:
        return None

    before_text = record["before"]
    after_text = record["after"]
    if not before_text:
        return None

    before_lines = split_lines(before_text)
    after_lines = split_lines(after_text)

    hunks = diff_hunks_text(before_lines, after_lines)
    if not hunks:
        return None

    task_marker = task_marker_fmt.format(TASK_TOKEN_NAME[ttype])
    enc_text = task_marker + before_text
    dec_text = build_sr_target_text(hunks, search_marker, replace_marker)

    if len(enc_text) > max_src_chars or len(dec_text) > max_tgt_chars:
        return None

    return enc_text, dec_text, TASK_ID[ttype]
