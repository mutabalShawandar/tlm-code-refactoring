import re
import io
import tokenize
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Python 3.12+ fragments f-strings into FSTRING_START/MIDDLE/END sub-tokens
# instead of one atomic STRING (3.11 and earlier). None on older interpreters.
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)

class TLM_Tokenizer:
    def __init__(self, vocab_size=10000):
        self.tokenizer = None
        self.vocab_size = vocab_size
        
        # Note: Do not use these sets for boundary checks against multi-char operators
        # like '+=' or '->', as it will shatter them into single characters.
        self.punctuation = set("(){}[],:=+-*/%<>!&|^~")
        self.structural_chars = set("(){}[],:.=+-*/%<>!&|^~\"'")

        # Placeholder for a literal space *inside* an atomic string/comment pretoken
        # (e.g. the space in `# Row` or `- 'args'` inside a docstring). BPE's
        # WhitespaceSplit pre-tokenizer splits on every whitespace run before BPE ever
        # sees it, so a real " " there is indistinguishable from the separator
        # `pretokenize()`'s `" ".join()` inserts between unrelated pretokens -- both
        # collapse to "no space" on decode. U+2581 ("▁") is not Unicode-whitespace, so
        # WhitespaceSplit never splits on it, letting it survive through BPE and get
        # mapped back to " " in decode(). Must stay in `_build_initial_alphabet()` so
        # BPE treats it as a base symbol instead of falling back to <unk>.
        # e.g. "# hello world" -> "#▁hello▁world" before joining with neighboring
        # pretokens, so WhitespaceSplit sees it as one chunk instead of shattering it
        # into "#", "hello", "world" indistinguishably from three unrelated pretokens.
        self.SPACE_MARKER = "▁"

        # Prefix marking "this pretoken is a continuation line of a multi-line
        # string/comment" -- distinct from SPACE_MARKER because a continuation line can
        # be flush-left (zero real leading whitespace, e.g. a deliberately unindented
        # section header inside an indented docstring) and still needs to suppress
        # decode()'s synthetic re-indentation. Stripped entirely on decode, never
        # rendered as a space. Zero-width so it's invisible if a stray one ever survives
        # into output.
        # e.g. without this, a flush-left docstring line inside an indented function
        # would get decode()'s synthetic 4-space indent added on top of its real
        # (zero) leading whitespace, corrupting the string's own formatting.
        self.CONTENT_MARKER = "​"
        
        self.special_tokens = {
            # Base Special Tokens
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "mask_token": "<mask>",
            "sep_token": "<sep>",
            
            # Python Structural Tokens
            "indent_token": "<indent>",
            "dedent_token": "<dedent>",
            "newline_token": "<newline>",
            
            # Refactoring Task Flags
            "rename_token": "<rename>",
            "simplify_token": "<simplify>",
            "deduplicate_token": "<deduplicate>",
            "remove_dead_code_token": "<remove_dead_code>",
            "extract_method_token": "<extract_method>",

            # Search/Replace
            "search_token": "<search>",
            "replace_token": "<replace>",
        }

        # Sentinel tokens for masking/denoising tasks
        for i in range(64):
            self.special_tokens[f"extra_id_{i}"] = f"<extra_id_{i}>"
            
        # Python Pre-tokenization Regex (Strictly ordered: longest operators first!)
        self.py_token_pattern = re.compile(
            # 1. Multi-char operators (Longest first to prevent greedy splitting)
            r"(?:\*\*=|//=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|==|!=|<=|>=|>>|<<|:=|\->)"
            # 2. Strings (with an optional string-prefix letter combo -- f/r/u/b and
            #    their 2-letter combos, e.g. rb/fr -- glued to the quote as ONE token.
            #    Must come before the identifier alternative below, or the prefix
            #    letter(s) get split off as their own token and `decode()` has no way
            #    to know it must not insert a space before the following quote (found
            #    via bridging generator round-trip testing: f'...'/r'...'/u'...' all
            #    decoded with a spurious space -- f 'foo' -- which is a syntax error).
            r"|(?:[bBfFrRuU]{1,2})?\"(?:\\.|[^\"\\])*\""
            r"|(?:[bBfFrRuU]{1,2})?\'(?:\\.|[^\'\\])*\'"
            # 3. Identifiers and keywords
            r"|[A-Za-z_]\w*"
            # 4. Numbers -- hex/octal/binary ints, floats/ints with a scientific-
            #    notation exponent, plain floats, plain ints. The original pattern
            #    (`\d+\.\d+|\d+`) had none of the exponent/hex/octal/binary cases,
            #    so e.g. `1e-7` split into "1"/"e"/"-"/"7" and `0x0009` split into
            #    "0"/"x0009" -- decode() then inserted spaces inside what should be
            #    one literal (`1 e-7`, `0 x0009`), breaking the syntax (found via
            #    bridging generator round-trip testing).
            r"|0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+"
            r"|\d+\.\d+(?:[eE][+-]?\d+)?[jJ]?|\d+[eE][+-]?\d+[jJ]?|\d+[jJ]?"
            # 5. Single-char punctuation and structural characters
            r"|[\(\)\[\]\{\},:.;@#~%^&*/\\+\-<>=|$]"
        )

    # --- setup helpers ---------------------------------------------------

    def _build_initial_alphabet(self):
        alphabet = set(self.structural_chars)
        alphabet.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        alphabet.update({" ", "\n", "\t", self.SPACE_MARKER, self.CONTENT_MARKER})
        return sorted(alphabet)

    def normalize_code(self, code: str) -> str:
        return code.replace("\r\n", "\n").replace("\t", "    ").strip() + "\n"

    # --- pretokenization: source text -> a flat list of pretokens --------
    # pretokenize() dispatches to one of the two implementations below,
    # then escapes interior spaces before BPE ever sees the result.

    def pretokenize(self, code):
        """Hybrid pretokenizer: stdlib `tokenize` (CPython's own lexer) is the
        primary path -- it correctly handles bracket-depth and backslash-
        continuation for free, since both are literal grammar behavior, not
        edge cases to special-case, on top of everything the regex tokenizer
        below fixed by hand (string prefixes, number literals, unicode
        identifiers, comment boundaries). Falls back to `_pretokenize_regex`
        (lenient, never raises) if stdlib `tokenize` can't lex the input.

        In practice this fallback is a defensive-only safety net, not a live
        code path: this project's pretraining corpus is CodeSearchNet, whose
        own extraction already yields complete, syntactically valid
        functions, so stdlib `tokenize` never actually fails on it. Measured
        (src/utils/hybrid_tokenize_check.py) at 0.76% -> 0.02% corpus
        decode-invalid rate, 0% fallback trigger across all 194,627 real
        snippets in the pretraining corpus."""
        try:
            tokens = self._pretokenize_stdlib(code)
        except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
            tokens = self._pretokenize_regex(code) ## Falback to regex-based pretokenizer on lexically invalid input
        return self._escape_interior_spaces(tokens)

    def _escape_interior_spaces(self, tokens):
        """Replace literal " " inside atomic pretokens (comments, string literals) with
        SPACE_MARKER so it survives BPE's WhitespaceSplit pre-tokenizer instead of being
        silently discarded (see SPACE_MARKER docstring above `__init__`). Special tokens
        (`<newline>`, `<indent>`, ...) never contain a space, so they pass through
        untouched -- only real source-text pretokens can be affected."""
        special_token_values = set(self.special_tokens.values())
        return [
            tok.replace(" ", self.SPACE_MARKER) if tok not in special_token_values and " " in tok else tok
            for tok in tokens
        ]

    def _pretokenize_stdlib(self, code):
        """Primary pretokenizer, built on stdlib `tokenize`. `code` must
        already be normalize_code()-d. Raises on lexically invalid input --
        caller (`pretokenize`) handles the lenient fallback.

        stdlib `tokenize` doesn't give two things this project needs, so both
        are reconstructed here: (1) a `<newline>` per *physical* source line
        -- `sft_corruption.py` diffs SEARCH/REPLACE hunks over `<newline>`-
        split id streams, but stdlib tokenize marks only *logical* line ends
        and returns a multi-line string as one token spanning several rows,
        so physical breaks (including inside an open bracket, where nothing
        is emitted at all) are re-derived from each token's row span via
        `fill_gap`/`emit_multiline_text`; (2) a trailing "\\" for backslash-
        continued lines -- stdlib tokenize treats it as a pure continuation
        marker and never emits a token for it, so without restoring it the
        reconstructed source loses the marker and looks like two statements
        instead of one continued line (found the hard way: the first version
        of this fix regressed a same-size sample worse than the regex
        tokenizer it was replacing)."""
        lines = code.splitlines(keepends=True)
        out = []
        current_row = 1

        # Closures rather than helper methods: both need read/write access to
        # `out` and `current_row` on every call, which stdlib tokenize's loop
        # below drives line by line -- passing that state around explicitly
        # would mean threading the same two arguments through every call site.

        def fill_gap(target_row):
            """Emit one <newline> (plus a trailing '\\' where the source line
            ends with a continuation backslash) for each physical row between
            current_row and target_row."""
            nonlocal current_row
            while current_row < target_row:
                row_text = lines[current_row - 1] if current_row - 1 < len(lines) else ""
                if row_text.rstrip("\r\n").endswith("\\"):
                    out.append("\\")
                out.append(self.special_tokens["newline_token"])
                current_row += 1

        def emit_multiline_text(text, start_row, end_row):
            """Split a multi-line STRING/f-string token into per-physical-line
            pretokens, inserting a <newline> between rows and flagging every
            row after the first as a preserved-content continuation line."""
            nonlocal current_row
            fill_gap(start_row)
            parts = text.split("\n")
            for idx, part in enumerate(parts):
                if part:
                    # idx 0 is the physical line the string *opens* on -- it shares
                    # that line with real code (e.g. `x = """...`), so the code's own
                    # <indent>/<newline> handling already positions it correctly.
                    # idx >= 1 is a pure continuation line living entirely inside the
                    # string: decode() has no other way to know it must not apply its
                    # synthetic code-indent here, so it's flagged with CONTENT_MARKER
                    # regardless of whether the line itself has real leading
                    # whitespace -- a flush-left continuation line (no leading spaces,
                    # e.g. a deliberately unindented "Output:" section header inside an
                    # indented docstring) would otherwise be indistinguishable from a
                    # real code token and get wrongly re-indented.
                    out.append((self.CONTENT_MARKER + part) if idx > 0 else part)
                if idx != len(parts) - 1:
                    out.append(self.special_tokens["newline_token"])
            current_row = end_row

        pending_fstring_start = None
        fstring_depth = 0

        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            ttype, tstring, (srow, scol), (erow, ecol), _ = tok

            if ttype in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue

            if _FSTRING_START is not None and ttype == _FSTRING_START:
                if fstring_depth == 0:
                    pending_fstring_start = (srow, scol)
                fstring_depth += 1
                continue
            if _FSTRING_START is not None and ttype == _FSTRING_MIDDLE:
                continue
            if _FSTRING_START is not None and ttype == _FSTRING_END:
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
                out.append(self.special_tokens["indent_token"])
                continue
            if ttype == tokenize.DEDENT:
                fill_gap(srow)
                out.append(self.special_tokens["dedent_token"])
                continue
            if ttype in (tokenize.NEWLINE, tokenize.NL):
                fill_gap(srow)
                out.append(self.special_tokens["newline_token"])
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

    def _pretokenize_regex(self, code):
        """Lenient fallback, never raises. Kept as a defensive safety net for
        lexically invalid input; not exercised in practice by this project's
        own corpus (see `pretokenize`'s docstring) but still load-bearing for
        any future corpus that isn't pre-extracted into complete functions.
        Line-by-line with a state machine for triple-quoted strings.

        Indentation is tracked as a stack of observed widths (like Python's
        own tokenizer/grammar), not by dividing the leading-space count by a
        fixed constant. A hardcoded `indent // 4` here previously silently
        produced zero <indent>/<dedent> tokens for any indentation width that
        wasn't a multiple of 4 (2-space, 3-space, etc. all collapsed to level
        0) -- found via bridging generator round-trip testing exposing
        invalid-Python reconstructions; confirmed present in the pretrain
        corpus and curated SFT dataset too (~10-17% and ~2-30% of examples
        respectively). A stack only needs each
        new indented block to use *more* leading whitespace than its
        enclosing block -- it never needs to know or assume the width."""
        tokens = []
        indent_stack = [0]
        in_ml_string = False
        ml_delim = None

        # Running paren/bracket/brace depth, updated only from tokens matched
        # in actual *code* regions (never from string content -- a stray "("
        # inside a string literal must not perturb this). While depth > 0
        # entering a physical line, that line is a continuation of the
        # previous logical statement (e.g. a multi-line call/def signature
        # inside an unclosed "("), so its leading whitespace is NOT block
        # indentation and must not be fed to indent_stack. Known unresolved
        # gap: this only tracks depth from tokens seen so far on the current
        # line, so a bracket opened and closed across several lines in an
        # unusual layout can still misclassify a line's indentation.
        bracket_depth = 0
        OPENERS = "([{"
        CLOSERS = ")]}"

        def emit_code(text):
            # emit_code is only ever called on a slice of a single physical
            # line, so once a bare "#" token is seen everything after it on
            # this call is a comment -- stop feeding tokens to bracket_depth
            # (a stray "[" or "(" inside a comment, e.g. an ANSI-escape note
            # like "# <ESC>[2K", would otherwise increment bracket_depth with
            # no matching closer anywhere in the file, silently disabling
            # indent/dedent tracking for everything after it -- found via the
            # same real-corpus round-trip check used for this bracket-depth
            # fix). A standalone "#" token can only come from real code,
            # never from inside a string, since string literals are already
            # matched whole by an earlier regex alternative.
            nonlocal bracket_depth
            in_comment = False
            for m in self.py_token_pattern.finditer(text):
                g = m.group(0)
                tokens.append(g)
                if in_comment:
                    continue
                if g == "#":
                    in_comment = True
                elif g in OPENERS:
                    bracket_depth += 1
                elif g in CLOSERS:
                    bracket_depth = max(0, bracket_depth - 1)

        for line in code.splitlines():
            if in_ml_string:
                stripped = line.strip()
                close_idx = stripped.find(ml_delim)
                if close_idx >= 0:
                    before = stripped[:close_idx]
                    after  = stripped[close_idx + 3:]
                    if before:
                        for m in self.py_token_pattern.finditer(before):
                            tokens.append(m.group(0))
                    tokens.append(ml_delim)
                    in_ml_string = False
                    ml_delim = None
                    if after:
                        emit_code(after)
                    tokens.append(self.special_tokens["newline_token"])
                else:
                    if stripped:
                        for m in self.py_token_pattern.finditer(stripped):
                            tokens.append(m.group(0))
                    tokens.append(self.special_tokens["newline_token"])
            else:
                if not line.strip():
                    tokens.append(self.special_tokens["newline_token"])
                    continue
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                if bracket_depth == 0:
                    if indent > indent_stack[-1]:
                        indent_stack.append(indent)
                        tokens.append(self.special_tokens["indent_token"])
                    else:
                        while indent < indent_stack[-1]:
                            indent_stack.pop()
                            tokens.append(self.special_tokens["dedent_token"])

                pos = 0
                while pos < len(stripped):
                    dq = stripped.find('"""', pos)
                    sq = stripped.find("'''", pos)
                    if dq == -1 and sq == -1:
                        emit_code(stripped[pos:])
                        pos = len(stripped)
                    else:
                        if dq == -1:   tq_pos, delim = sq, "'''"
                        elif sq == -1: tq_pos, delim = dq, '"""'
                        else:          tq_pos, delim = (dq, '"""') if dq < sq else (sq, "'''")
                        if tq_pos > pos:
                            emit_code(stripped[pos:tq_pos])
                        rest = stripped[tq_pos + 3:]
                        close_in_rest = rest.find(delim)
                        if close_in_rest >= 0:
                            content = rest[:close_in_rest]
                            tokens.append(delim)
                            if content:
                                for m in self.py_token_pattern.finditer(content):
                                    tokens.append(m.group(0))
                            tokens.append(delim)
                            pos = tq_pos + 3 + close_in_rest + 3
                        else:
                            tokens.append(delim)
                            if rest.strip():
                                for m in self.py_token_pattern.finditer(rest):
                                    tokens.append(m.group(0))
                            tokens.append(self.special_tokens["newline_token"])
                            in_ml_string = True
                            ml_delim = delim
                            pos = len(stripped)

                if not in_ml_string:
                    tokens.append(self.special_tokens["newline_token"])

        while len(indent_stack) > 1:
            indent_stack.pop()
            tokens.append(self.special_tokens["dedent_token"])
        return tokens

    # --- BPE training, encode, decode -------------------------------------

    def train(self, codes):
        special_token_list = list(self.special_tokens.values())
        initial_alphabet = self._build_initial_alphabet()

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_token_list,
            initial_alphabet=initial_alphabet,
            end_of_word_suffix="</w>",
        )

        processed_codes = []
        for code in codes:
            normalized = self.normalize_code(code)
            tokens = self.pretokenize(normalized)
            processed_codes.append(" ".join(tokens))

        tokenizer.train_from_iterator(processed_codes, trainer=trainer)
        self.tokenizer = tokenizer

        print(f"Trained code tokenizer with vocab_size={self.vocab_size}")
        print("Special token verification:")
        for name, tok in self.special_tokens.items():
            tid = self.tokenizer.token_to_id(tok)
            back = self.tokenizer.id_to_token(tid)
            print(f"  {name}: '{tok}' -> id {tid} -> '{back}'")

    def encode(self, code: str) -> list:
        norm = self.normalize_code(code)
        tokens = self.pretokenize(norm)
        enc = self.tokenizer.encode(" ".join(tokens))
        return enc.ids

    def decode(self, token_ids, debug: bool = False) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet. Call train() first.")

        tokens = [self.tokenizer.id_to_token(tid) for tid in token_ids]
        special_token_values = set(self.special_tokens.values())
        # single-char punctuation and quote delimiters -- no space on either side.
        # NOT "_": mid-word BPE fragments (e.g. joining "some"+"_"+"name") are
        # already handled separately by the word_buffer accumulation below and
        # never consult this set at all; this set is only consulted for a
        # *complete* token, and "_" is frequently a complete, standalone
        # identifier on its own (Python's throwaway-variable convention) that
        # needs normal spacing like any other word -- treating it as punctuation
        # glued "for", "_", "in" into "for_in" (found via bridging generator
        # round-trip testing).
        # single-char alternatives here only ever match a *single*
        # character, so a whole multi-char operator token (e.g. "**", "==", "->") is
        # never equal to any set member -- `"**" in set("...*...")` is False even
        # though `"*"` is in the set. That left every multi-char operator out of the
        # no-trailing-space rule, so e.g. `**kwargs` decoded as `** kwargs`. Listed
        # explicitly here as whole-string entries alongside the single-char set.
        multi_char_operators = {
            "**", "//", "<<", ">>", "==", "!=", "<=", ">=", "->", ":=",
            "+=", "-=", "*=", "/=", "//=", "**=", "%=", "&=", "|=", "^=", ">>=", "<<=", "@=",
        }
        punctuation_tokens = set("(){}[],:=+-*/%<>!&|^~.#@") | {"'''", '"""'} | multi_char_operators

        output = []
        indent_level = 0
        word_buffer = []  # accumulates BPE fragments until </w> is seen

        # Closure over `tokens`/`output` for the same reason as fill_gap/
        # emit_multiline_text above: it needs to look ahead at the next raw
        # token to decide on spacing, and append straight to `output`.
        def emit_word(word, next_idx):
            """Append word to output with a trailing space iff needed."""
            next_tok = tokens[next_idx] if next_idx < len(tokens) else None
            next_is_special = (next_tok is None or next_tok in special_token_values)
            curr_is_punct = (word in punctuation_tokens)
            next_first = None
            if next_tok is not None and not next_is_special:
                # next_tok is either a fragment or a complete </w> word
                next_first = next_tok[:-4] if next_tok.endswith("</w>") else next_tok
                next_is_punct = (next_tok.endswith("</w>") and next_first in punctuation_tokens)
            else:
                next_is_punct = False
            # Whether real source whitespace preceded a closing '''/""" is not
            # recoverable at decode time in general (WhitespaceSplit discards
            # exact spacing before BPE ever sees it, and pretokenize's uniform
            # " ".join() already fakes a separator between every regular
            # punctuation token regardless of true source adjacency) -- so the
            # blanket "no space before '''/\"\"\"" rule is a deliberate, mostly-
            # correct guess (closing delimiters overwhelmingly abut their
            # content with no gap, e.g. "done here.\"\"\""). It only breaks when
            # the content word itself ends in a literal quote character: then
            # dropping the (possibly real) gap merges into an ambiguous run of
            # 4 quote chars, invalid string syntax (found via the hybrid-
            # tokenizer corpus round-trip check -- e.g. "'a_list'" + "'''" ->
            # "'a_list''''"). Force a space only for that specific case; the
            # `word not in ("'''", '"""')` guard keeps an immediately-adjacent
            # empty string ("''''''") from picking up a spurious space.
            force_space = (
                next_is_punct and next_first in ("'''", '"""')
                and word not in ("'''", '"""')
                and word.endswith(("'", '"'))
            )
            add_space = force_space or not (curr_is_punct or next_is_special or next_is_punct)
            display_word = word.replace(self.CONTENT_MARKER, "").replace(self.SPACE_MARKER, " ")
            output.append(display_word + " " if add_space else display_word)

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok is None:
                i += 1
                continue

            if tok == self.special_tokens["newline_token"]:
                if word_buffer:
                    output.append("".join(word_buffer).replace(self.CONTENT_MARKER, "").replace(self.SPACE_MARKER, " "))
                    word_buffer = []
                j = i + 1
                while j < len(tokens) and tokens[j] in (
                    self.special_tokens["indent_token"],
                    self.special_tokens["dedent_token"],
                ):
                    if tokens[j] == self.special_tokens["indent_token"]:
                        indent_level += 1
                    else:
                        indent_level = max(0, indent_level - 1)
                    j += 1
                # A continuation line of a multi-line string/comment
                # carries its own real leading whitespace verbatim (escaped with
                # SPACE_MARKER, restored on emit -- see _escape_interior_spaces). Adding
                # the synthetic "    "*indent_level here on top of that would double the
                # indentation (confirmed via corpus ast.dump comparison: a docstring
                # indented 8 spaces in the source came back indented 12). Such a line is
                # never a real code statement -- stdlib tokenize's INDENT/DEDENT tokens
                # already carry the block-structure indentation, so it's flagged with
                # CONTENT_MARKER (see emit_multiline_text) unambiguously meaning "don't
                # add synthetic indentation here."
                # A blank *interior* line of a multi-line string produces no content
                # token at all (see `_pretokenize_stdlib.emit_multiline_text`: an empty
                # `part` is never appended) -- just another back-to-back <newline>. So
                # peeking only one token ahead would land on that next <newline> (a
                # special token) rather than the real content, wrongly concluding "not
                # preserved content" and re-indenting what should be a truly blank line.
                # Skip forward through any run of <newline>/<indent>/<dedent> to find the
                # actual next substantive token before deciding.
                peek_j = j
                while peek_j < len(tokens) and tokens[peek_j] in (
                    self.special_tokens["newline_token"],
                    self.special_tokens["indent_token"],
                    self.special_tokens["dedent_token"],
                ):
                    peek_j += 1
                next_tok = tokens[peek_j] if peek_j < len(tokens) else None
                next_is_preserved_content = (
                    next_tok is not None
                    and next_tok not in special_token_values
                    and next_tok.startswith(self.CONTENT_MARKER)
                )
                indent_str = "" if next_is_preserved_content else "    " * indent_level
                output.append(("\n" + indent_str + "[<newline>]") if debug
                               else ("\n" + indent_str))
                i = j
                continue

            if tok in special_token_values:
                if word_buffer:
                    output.append("".join(word_buffer).replace(self.CONTENT_MARKER, "").replace(self.SPACE_MARKER, " "))
                    word_buffer = []
                if debug:
                    output.append(f"[{tok}]")
                i += 1
                continue

            if tok.endswith("</w>"):
                word_buffer.append(tok[:-4])
                word = "".join(word_buffer)
                word_buffer = []
                emit_word(word, i + 1)
            else:
                # mid-word BPE fragment -- accumulate without spacing
                word_buffer.append(tok)

            i += 1

        if word_buffer:
            output.append("".join(word_buffer).replace(self.CONTENT_MARKER, "").replace(self.SPACE_MARKER, " "))

        return "".join(output).strip()

    # --- accessors ---------------------------------------------------------

    def id_to_token(self, token_id):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        return self.tokenizer.id_to_token(token_id)
    
    def token_to_id(self, token):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        return self.tokenizer.token_to_id(token)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        
       
