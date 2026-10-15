

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import re
import keyword
import hashlib
import json
# -- Config and Paths ------------------------------
CODE_KEYS = [
    "code", "snippet", "text", "content", "source", "python",
    "function", "original", "input", "raw_code", "completion", "body",
]


SPLIT_RATIOS = {"train": 0.90, "val": 0.05, "test": 0.05}

# -- Identifier detection ------------------------------------------------------
PYTHON_KEYWORDS = frozenset(keyword.kwlist)
IDENTIFIER_RE   = re.compile(r'^[A-Za-z_]\w{1,}$')  # length >= 2 (skip i, x, n)
TRIPLE_QUOTES   = frozenset({"'''", '"""'})


# -- Section A: Code extraction & deterministic train/val split ----------------

def extract_code(record: dict) -> str:
    """Pick the code string out of a JSONL record regardless of field name."""
    for key in CODE_KEYS:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""




def _split_hash(record: dict, code: str) -> float:
    """Stable [0, 1) float derived from repo/file identity -- same as XS split."""
    anchor = (
        record.get("repo_hash")
        or record.get("file_hash")
        or record.get("url")
        or code[:200]
    )
    hex_key = hashlib.md5(anchor.encode()).hexdigest()
    return int(hex_key[:8], 16) / 0xFFFFFFFF


def split_name_for_record(record: dict, code: str) -> str:
    v = _split_hash(record, code)
    if v < SPLIT_RATIOS["train"]:
        return "train"
    if v < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def iter_raw_code(path: Path, split_name: Optional[str] = None):
    """Yield raw code strings from a tier JSONL, optionally filtered by split."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                code = extract_code(record)
                if not code.strip():
                    continue
                if split_name is None or split_name_for_record(record, code) == split_name:
                    yield code
            except Exception:
                continue


def pretokenize_with_docstring_mask(tok, code: str) -> Tuple[List[str], List[bool]]:
    """
    Returns (pretokens, in_docstring) where in_docstring[i] is True if pretokens[i]
    is inside a triple-quoted string.  Uses the tokenizer's own pretokenize() so the
    index mapping to BPE tokens is exact.
    """
    pretokens = tok.pretokenize(tok.normalize_code(code))
    in_ds = []
    inside = False
    delim  = None
    for pt in pretokens:
        if not inside:
            if pt in TRIPLE_QUOTES:
                inside = True
                delim  = pt
                in_ds.append(True)   # the opening delimiter itself is in docstring
            else:
                in_ds.append(False)
        else:
            in_ds.append(True)
            if pt == delim:          # closing delimiter
                inside = False
                delim  = None
    return pretokens, in_ds


def pretoken_to_bpe_ranges(tok, pretokens: List[str]) -> List[Tuple[int, int]]:
    """
    Map each pre-token index -> (bpe_start, bpe_end) exclusive using HF word_ids.
    Uses tokenizer.tokenizer (the underlying HF Tokenizer) directly so we get
    word_ids without wrapping.

    NOTE: encode() joins pretokens with spaces and the WhitespaceSplit pre-tokenizer
    re-splits on whitespace.  A single pretoken can contain internal spaces (e.g. a
    string literal '"a b c"'), so it expands to several HF words.  We therefore map
    HF word index -> pretoken index via cumulative per-pretoken word counts
    (len(pt.split())), NOT the identity wid == pretoken_index -- which both crashes
    with IndexError and silently misaligns every span after such a literal.
    """
    joined = " ".join(pretokens)
    enc = tok.tokenizer.encode(joined)          # HF Encoding
    word_ids = enc.word_ids                     # list[int|None], one per BPE token
    # HF word index -> originating pretoken index
    word_to_pt: List[int] = []
    for pt_idx, pt in enumerate(pretokens):
        word_to_pt.extend([pt_idx] * max(1, len(pt.split())))
    ranges: List[Tuple[int, int]] = [None] * len(pretokens)
    for bpe_idx, wid in enumerate(word_ids):
        if wid is None or wid >= len(word_to_pt):
            continue
        pt = word_to_pt[wid]
        if ranges[pt] is None:
            ranges[pt] = (bpe_idx, bpe_idx + 1)
        else:
            ranges[pt] = (ranges[pt][0], bpe_idx + 1)
    # fill any None (special tokens not assigned a word) with zero-width sentinel
    return [(r if r is not None else (0, 0)) for r in ranges]


def find_identifier_bpe_spans(
    tok, pretokens: List[str], in_docstring: List[bool]
) -> List[Tuple[int, int]]:
    """
    Return BPE-level (start, end) spans for each maskable identifier:
      - matches IDENTIFIER_RE
      - not a Python keyword
      - not inside a docstring / triple-quoted string
      - not a special token (starts with '<')
    """
    pt_ranges = pretoken_to_bpe_ranges(tok, pretokens)
    spans = []

    for i, pt in enumerate(pretokens):
        if in_docstring[i]:
            continue
        if not IDENTIFIER_RE.match(pt):
            continue
        if pt in PYTHON_KEYWORDS:
            continue
        if pt.startswith("<"):          # special token string
            continue
        start, end = pt_ranges[i]
        if end <= start:
            continue
        spans.append((start, end))
    return spans

