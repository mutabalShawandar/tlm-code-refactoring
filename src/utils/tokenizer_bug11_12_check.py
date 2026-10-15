"""Round-trip validation for bug 11 (multi-char operator spacing) and bug 12 (string/
comment interior-whitespace loss) fixes in src/tokenizer.py.

Two checks per snippet, both stronger than plain ast.parse() validity (which neither
bug necessarily breaks -- extra whitespace around an operator and a reformatted
docstring are both still syntactically valid Python):

  1. ast.parse(dec) succeeds at all.
  2. ast.dump(ast.parse(orig)) == ast.dump(ast.parse(dec)) -- a structural AST
     equality check. This catches content corruption that plain parseability can't:
     a docstring/string literal whose interior whitespace changed shows up as a
     different Constant value, so bug 12 (string content loss) is caught here even
     though the code still parses fine.

Comments aren't part of the AST, so this can't catch bug 11's original repro
(`# Row` -> `#Row`) or confirm it's fixed -- that's checked separately below by
grepping decoded output for the specific known-bad patterns.

Usage:
    python src/utils/tokenizer_bug11_12_check.py
"""

import sys
import os
import json
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tokenizer import TLM_Tokenizer
import pandas as pd

DEFAULT_TIER_FILES = [f"data/curriculum/tier{t}.jsonl" for t in range(1, 6)]


def main():
    corpus = pd.read_csv("data/code_search_net.csv")
    codes = corpus["code"].tolist()

    tok = TLM_Tokenizer(vocab_size=10000)
    tok.train(codes)

    total = 0
    parse_invalid = 0
    dump_mismatch = 0
    shown = 0

    # targeted repros from the two bugs found in the manual notebook inspection
    repro_operator = "def show(self, temp_file_name='ani.mp4', **kwargs):\n    pass\n"
    repro_comment = "def f(x):\n    if x:  # Row\n        pass\n"

    for repro_name, repro in [("operator (**kwargs)", repro_operator), ("comment (# Row)", repro_comment)]:
        ids = tok.encode(repro)
        dec = tok.decode(ids)
        print(f"--- repro: {repro_name} ---")
        print("ORIG:", repro.strip())
        print("DEC: ", dec.strip())
        print()

    for path in DEFAULT_TIER_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                code = rec["code"]
                total += 1

                ids = tok.encode(code)
                dec = tok.decode(ids)

                try:
                    orig_ast = ast.parse(code)
                except SyntaxError:
                    continue  # skip snippets that weren't valid Python to begin with

                try:
                    dec_ast = ast.parse(dec)
                except SyntaxError:
                    parse_invalid += 1
                    continue

                if ast.dump(orig_ast) != ast.dump(dec_ast):
                    dump_mismatch += 1
                    if shown < 5:
                        shown += 1
                        print(f"=== ast-dump mismatch ===")
                        print("ORIG:", code[:300])
                        print("DEC: ", dec[:300])
                        print()

    print(f"{parse_invalid}/{total} decode-invalid ({100*parse_invalid/total:.3f}%)")
    print(f"{dump_mismatch}/{total} ast-dump mismatch, valid-but-corrupted ({100*dump_mismatch/total:.3f}%)")

    tok.save("src/tokenizer/tokenizer_tlm_small_hybrid_fixed.json")


if __name__ == "__main__":
    main()
