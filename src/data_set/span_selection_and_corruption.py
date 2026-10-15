# -- Section B: Span selection + SEARCH/REPLACE corruption --------------------

from typing import Dict, Optional, List, Tuple, Any
import random
from .pretrain_loaders import (
    pretokenize_with_docstring_mask,
    pretoken_to_bpe_ranges,
    find_identifier_bpe_spans
    )

# -- Spanning constraints ------------------------------------------------------
SPAN_MASK_PROB = 0.12    
SPAN_AVG_LEN   = 4      
MAX_MASK_SPANS = 5      
MIP_PROB       = 0.45   
NUM_SENTINELS  = 64     

MAX_SRC_LEN = 512       
MAX_TGT_LEN = 512       
                        



def _build_line_map(bpe_ids: List[int], newline_id: int) -> List[int]:
    """Map each BPE position -> 0-indexed line number (lines delimited by newline_id)."""
    line_map = []
    cur = 0
    for tid in bpe_ids:
        line_map.append(cur)
        if tid == newline_id:
            cur += 1
    return line_map


def choose_spans(
    bpe_ids: List[int],
    in_docstring_bpe: List[bool],
    identifier_spans: List[Tuple[int, int]],
    newline_id: int,
) -> List[Tuple[int, int]]:
    """
    Select BPE-level spans to mask.  Constraints:
      - No span crosses a <newline> token
      - No two spans overlap
      - At most one span per line (keeps each SEARCH block unambiguous)
      - At most MAX_MASK_SPANS spans; total masked tokens <= SPAN_MASK_PROB * N
    Mode per example: MIP (identifier spans) with prob MIP_PROB, else random geometric.
    Falls back to random if no MIP candidates survive the docstring/bounds filters.
    """
    N = len(bpe_ids)
    if N == 0:
        return []
    budget   = max(1, int(SPAN_MASK_PROB * N))
    line_map = _build_line_map(bpe_ids, newline_id)

    use_mip = random.random() < MIP_PROB and bool(identifier_spans)

    if use_mip:
        candidates = [
            (s, e) for s, e in identifier_spans
            if 0 <= s < e <= N
            and not any(in_docstring_bpe[p] for p in range(s, e))
        ]
        if not candidates:
            use_mip = False  # fall back to random

    if use_mip:
        random.shuffle(candidates)
        selected: List[Tuple[int, int]] = []
        masked: set = set()
        masked_lines: set = set()
        tokens_masked = 0
        for s, e in candidates:
            if any(p in masked for p in range(s, e)):
                continue
            ln = line_map[s]
            if ln in masked_lines:
                continue
            if tokens_masked + (e - s) > budget:
                continue
            selected.append((s, e))
            masked.update(range(s, e))
            masked_lines.add(ln)
            tokens_masked += (e - s)
            if len(selected) >= MAX_MASK_SPANS:
                break
        return sorted(selected)

    # -- random geometric spans ------------------------------------------------
    valid = [
        i for i in range(N)
        if bpe_ids[i] != newline_id and not in_docstring_bpe[i]
    ]
    if not valid:
        return []

    selected      = []
    masked        = set()
    masked_lines  = set()
    tokens_masked = 0
    attempts      = 0

    while (len(selected) < MAX_MASK_SPANS
           and tokens_masked < budget
           and attempts < MAX_MASK_SPANS * 6):
        attempts += 1
        start = random.choice(valid)
        if start in masked:
            continue
        ln = line_map[start]
        if ln in masked_lines:
            continue
        # Geometric extension: stop prob = 1/SPAN_AVG_LEN per step
        end = start + 1
        while (end < N
               and bpe_ids[end] != newline_id
               and end not in masked
               and not in_docstring_bpe[end]
               and random.random() > 1.0 / SPAN_AVG_LEN):
            end += 1
        span_len = end - start
        if tokens_masked + span_len > budget:
            attempts += 1
            continue
        selected.append((start, end))
        masked.update(range(start, end))
        masked_lines.add(ln)
        tokens_masked += span_len

    return sorted(selected)


def make_sr_corruption(
    code: str, tok, cfg
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Build a SEARCH/REPLACE training pair from raw code:
      enc: <bos> [code with sentinel tokens replacing masked spans] <eos>
      dec: <bos> (<search> [corrupted_line] <replace> [original_line])* <eos>
    One SEARCH/REPLACE pair per span, in source order.
    Returns (enc_ids, dec_ids) or None if no spans found or output exceeds length limits.
    """
    NEWLINE_ID = cfg.special_token_ids["newline"]
    BOS_ID     = cfg.special_token_ids["bos"]
    EOS_ID     = cfg.special_token_ids["eos"]
    SEARCH_ID  = cfg.special_token_ids["search"]
    REPLACE_ID = cfg.special_token_ids["replace"]

    # Pretokenize + docstring mask
    pretokens, in_ds_pt = pretokenize_with_docstring_mask(tok, code)
    if not pretokens:
        return None

    # BPE content ids (no bos/eos -- tok.encode never adds them)
    content_ids: List[int] = tok.encode(code)
    N = len(content_ids)
    if N == 0:
        return None

    # Propagate pretoken-level docstring mask to BPE positions via word_id ranges
    pt_ranges = pretoken_to_bpe_ranges(tok, pretokens)
    in_ds_bpe = [False] * N
    for i, (s, e) in enumerate(pt_ranges):
        if in_ds_pt[i]:
            for j in range(s, min(e, N)):
                in_ds_bpe[j] = True

    # Identifier spans (MIP candidates)
    id_spans = find_identifier_bpe_spans(tok, pretokens, in_ds_pt)

    # Choose which BPE spans to mask
    spans = choose_spans(content_ids, in_ds_bpe, id_spans, NEWLINE_ID)
    if not spans:
        return None

    # Build corrupted encoder content: replace each span with its sentinel token
    enc_content: List[int] = []
    prev = 0
    for i, (s, e) in enumerate(spans):
        enc_content.extend(content_ids[prev:s])
        enc_content.append(cfg.special_token_ids[f"extra_id_{i}"])
        prev = e
    enc_content.extend(content_ids[prev:])
    enc_ids = [BOS_ID] + enc_content + [EOS_ID]

    # Build SEARCH/REPLACE decoder target.
    # One span per line is enforced by choose_spans, so each corrupted_line
    # contains exactly one sentinel and SEARCH matches what the encoder sees.
    dec_body: List[int] = []
    for i, (s, e) in enumerate(spans):
        # Locate the line containing this span in original content_ids.
        # Scan back to the previous <newline> (or start); scan forward to the next.
        line_start = 0
        for k in range(s - 1, -1, -1):
            if content_ids[k] == NEWLINE_ID:
                line_start = k + 1
                break
        line_end = N
        for k in range(e, N):
            if content_ids[k] == NEWLINE_ID:
                line_end = k
                break

        original_line  = content_ids[line_start:line_end]
        sentinel_id    = cfg.special_token_ids[f"extra_id_{i}"]
        corrupted_line = (original_line[:s - line_start]
                          + [sentinel_id]
                          + original_line[e - line_start:])

        dec_body += [SEARCH_ID] + corrupted_line + [REPLACE_ID] + original_line

    dec_ids = [BOS_ID] + dec_body + [EOS_ID]

    if len(enc_ids) > MAX_SRC_LEN or len(dec_ids) > MAX_TGT_LEN:
        return None

    return enc_ids, dec_ids


