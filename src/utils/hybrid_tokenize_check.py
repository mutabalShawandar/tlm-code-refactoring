"""Prototype validation for a hybrid pretokenizer built on stdlib `tokenize`.

Idea:
  - stdlib `tokenize` as the primary lexer -- correctly handles bracket-depth (bug 7) and
    backslash-continuation (bug 8) continuation lines for free, since that's literal CPython
    grammar behavior, plus every other bug fixed by hand in src/tokenizer.py's regex approach
    (string prefixes, number literals, unicode identifiers, comment boundaries).
  - a physical-line resynthesis pass on top, since stdlib tokenize emits nothing for a
    continuation-line break inside an open bracket, and returns a multi-line triple-quoted
    string as ONE token spanning several rows -- both cases need re-splitting into our own
    <newline> markers to preserve the per-physical-line invariant sft_corruption.py's
    line-level SR diffing depends on.
  - a lenient fallback to the existing TLM_Tokenizer.pretokenize() regex/state-machine for any
    snippet stdlib tokenize can't lex (corpus snippets are extracted fragments, not always
    lexically self-contained -- unbalanced brackets/indentation raise there).
  - an f-string re-glue pass, since Python 3.12+ fragments an f-string into
    FSTRING_START/MIDDLE/END sub-tokens instead of one atomic STRING like 3.11 and earlier;
    re-composed here from the original source span so f-strings stay one atomic pretoken,
    matching how the existing regex tokenizer already treats them.

This is validation-only -- NOT wired into src/tokenizer.py. Measures, via the same real-corpus
ast.parse round-trip check used for every prior bug, whether this beats the current
bugs-1-7-fixed baseline (0.76% decode-invalid, 194,627 snippets) and how often the lenient
fallback triggers.

Usage:
    python src/utils/hybrid_tokenize_check.py
    python src/utils/hybrid_tokenize_check.py --n-per-tier 3000
    python src/utils/hybrid_tokenize_check.py --show 3
"""

import sys
import os
import io
import json
import ast
import argparse
import tokenize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tokenizer import TLM_Tokenizer

FSTRING_START = getattr(tokenize, "FSTRING_START", None)
FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)
FSTRING_END = getattr(tokenize, "FSTRING_END", None)

DEFAULT_TIER_FILES = [f"data/curriculum/tier{t}.jsonl" for t in range(1, 6)]


def _stdlib_pretokenize(code, special_tokens):
    """Returns a flat list of pretoken strings for already-normalized `code`, or raises
    (tokenize.TokenError / IndentationError / SyntaxError / ValueError) on lexically
    invalid input -- caller is responsible for the lenient fallback."""
    lines = code.splitlines(keepends=True)
    out = []
    current_row = 1

    def fill_gap(target_row):
        # Rows skipped entirely here are, by construction, rows tokenize gave us zero
        # events for -- either bracket-continuation (nothing to restore) or backslash-
        # continuation. stdlib tokenize treats a trailing "\" as a pure continuation
        # marker and never emits a token for it, so it's otherwise silently dropped from
        # the reconstructed source -- decode() then shows two statements with a <newline>
        # between them and no "\", which is invalid syntax (confirmed via the corpus
        # round-trip check: every one of the first-run mismatches was exactly this).
        # Re-inserting the literal "\" from the original source line before the
        # synthesized <newline> fixes it.
        nonlocal current_row
        while current_row < target_row:
            row_text = lines[current_row - 1] if current_row - 1 < len(lines) else ""
            if row_text.rstrip("\r\n").endswith("\\"):
                out.append("\\")
            out.append(special_tokens["newline_token"])
            current_row += 1

    def emit_multiline_text(text, start_row, end_row):
        nonlocal current_row
        fill_gap(start_row)
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if part:
                out.append(part)
            if i != len(parts) - 1:
                out.append(special_tokens["newline_token"])
        current_row = end_row

    pending_fstring_start = None
    fstring_depth = 0

    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        ttype, tstring, (srow, scol), (erow, ecol), _ = tok

        if ttype in (tokenize.ENCODING, tokenize.ENDMARKER):
            continue

        if FSTRING_START is not None and ttype == FSTRING_START:
            if fstring_depth == 0:
                pending_fstring_start = (srow, scol)
            fstring_depth += 1
            continue
        if FSTRING_START is not None and ttype == FSTRING_MIDDLE:
            continue
        if FSTRING_START is not None and ttype == FSTRING_END:
            fstring_depth -= 1
            if fstring_depth == 0 and pending_fstring_start is not None:
                start_row, start_col = pending_fstring_start
                if start_row == erow:
                    text = lines[start_row - 1][start_col:ecol]
                else:
                    first = lines[start_row - 1][start_col:]
                    middle = "".join(lines[start_row:erow - 1])
                    last = lines[erow - 1][:ecol]
                    text = first + middle + last
                emit_multiline_text(text, start_row, erow)
                pending_fstring_start = None
            continue
        if fstring_depth > 0:
            continue

        if ttype == tokenize.INDENT:
            fill_gap(srow)
            out.append(special_tokens["indent_token"])
            continue
        if ttype == tokenize.DEDENT:
            fill_gap(srow)
            out.append(special_tokens["dedent_token"])
            continue
        if ttype in (tokenize.NEWLINE, tokenize.NL):
            fill_gap(srow)
            out.append(special_tokens["newline_token"])
            current_row = srow + 1
            continue
        if ttype == tokenize.COMMENT:
            fill_gap(srow)
            out.append(tstring)
            current_row = srow
            continue
        if ttype == tokenize.STRING:
            if erow == srow:
                fill_gap(srow)
                out.append(tstring)
                current_row = erow
            else:
                emit_multiline_text(tstring, srow, erow)
            continue
        if ttype in (tokenize.NAME, tokenize.NUMBER, tokenize.OP):
            fill_gap(srow)
            out.append(tstring)
            current_row = erow
            continue
        # rare/other token types -- just track row position, emit nothing
        fill_gap(srow)
        current_row = erow

    return out


def hybrid_pretokenize(normalized_code, special_tokens, legacy_fn):
    """Returns (pretoken_list, used_stdlib: bool)."""
    try:
        return _stdlib_pretokenize(normalized_code, special_tokens), True
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return legacy_fn(normalized_code), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-tier", type=int, default=3000)
    parser.add_argument("--tokenizer", default="src/tokenizer/tokenizer_tlm_small.json")
    parser.add_argument("--show", type=int, default=0,
                         help="print up to N mismatch examples")
    args = parser.parse_args()

    tok = TLM_Tokenizer()
    tok.load(args.tokenizer)

    total = 0
    bad = 0
    fallback_count = 0
    shown = 0

    for path in DEFAULT_TIER_FILES:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.n_per_tier:
                    break
                rec = json.loads(line)
                code = rec["code"]
                total += 1

                norm = tok.normalize_code(code)
                pretoks, used_stdlib = hybrid_pretokenize(norm, tok.special_tokens, tok.pretokenize)
                if not used_stdlib:
                    fallback_count += 1

                enc = tok.tokenizer.encode(" ".join(pretoks))
                dec = tok.decode(enc.ids)
                try:
                    ast.parse(dec)
                except SyntaxError:
                    bad += 1
                    if shown < args.show:
                        shown += 1
                        print(f"=== mismatch (fallback={not used_stdlib}) ===")
                        print("ORIG:", code[:300])
                        print("DEC: ", dec[:300])
                        print()

    rate = 100 * bad / total if total else 0.0
    fb_rate = 100 * fallback_count / total if total else 0.0
    print(f"{bad}/{total} decode-invalid ({rate:.2f}%) -- baseline (bugs 1-7 fixed): 0.76%")
    print(f"lenient fallback triggered: {fallback_count}/{total} ({fb_rate:.2f}%)")


if __name__ == "__main__":
    main()
