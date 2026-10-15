import torch

MAX_NEW_TOKENS = 512  # hard rule: 300 -> ~24% truncated outputs, never lower this
TOKEN_REPEAT_WINDOWS = (1, 2, 3)  # token-cycle period to check for a runaway loop
MIN_TOKEN_REPEAT_OCCURRENCES = {1: 6, 2: 4, 3: 3}  # consecutive full-cycle repeats
# required per period before treating it as degenerate, independent of the
# line-repeat guard below (which only checks at newline boundaries). Originally
# shipped as a period-1-only check (single identical token repeating, e.g. a
# single BPE token for `email_` recurring hundreds of times with no newline in
# between) -- found via the rename failure dump
# (project_rename_failure_breakdown). That version silently never
# fired on the *actual* observed runaway loops: confirmed via the tokenizer
# vocab that `email_` splits into two separate tokens ("email", "_"), so the
# real pattern is a period-2 alternating cycle, not a single repeated token --
# a period-1-only check can never trip on it since no token ever equals the one
# immediately before it. Generalized to check periods 1-3, same shape as
# LINE_REPEAT_WINDOWS/_lines_repeat_n_times below but at token granularity
# (fires regardless of newlines, unlike the line-level guard). Higher
# occurrence counts for smaller periods (6 for period-1) since short windows
# are more likely to coincidentally match in legitimate code.
MIN_REPEAT_OCCURRENCES = 3  # require a k-line window to recur this many times
# consecutively (not just once) before treating it as a genuine degenerate loop.
# Found via the eval notebook's 05e round-trip: simplify samples
# truncating at k=1/k=2 all terminated cleanly at a real <eos> within a couple
# hundred tokens when the guard was disabled -- the model was transiently
# repeating a short chunk (e.g. `else:\n    return False`) twice while
# "unrolling" an if/elif/else chain, then converging on the right answer, not
# looping forever. A single duplication (2 occurrences) isn't enough evidence
# of a real loop; this also matches the guard's own original design intent
# (see docstrings below -- extract_method's genuine failure case repeats 3x+).


def _lines_repeat_n_times(hunk_lines, k, n):
    """True if the last n non-overlapping k-line windows are all identical."""
    if len(hunk_lines) < n * k:
        return False
    windows = [tuple(hunk_lines[len(hunk_lines) - (i + 1) * k: len(hunk_lines) - i * k])
               for i in range(n)]
    return any(windows[0]) and len(set(windows)) == 1


def _tokens_repeat_n_times(tok_list, k, n):
    """True if the last n non-overlapping k-token windows are all identical.
    Same shape as _lines_repeat_n_times, generalized to any period k (1-3),
    checked against the raw token stream rather than completed lines -- catches
    runaway cycles that never hit a newline (see TOKEN_REPEAT_WINDOWS docstring
    above)."""
    if len(tok_list) < n * k:
        return False
    windows = [tuple(tok_list[len(tok_list) - (i + 1) * k: len(tok_list) - i * k])
               for i in range(n)]
    return len(set(windows)) == 1


def _mask_banned_ngrams(log_probs_t, generated_tail, banned_ngrams):
    """Set the logit of any candidate token to -inf if choosing it would complete
    one of `banned_ngrams` (each a tuple of token ids) at the tail of the
    sequence-so-far. Only clones `log_probs_t` if a mask actually applies.

    Used for the rename name-collision repair pass (post-hoc detect + re-decode,
    see project_rename_failure_breakdown / find_pred_collisions in
    src/eval/scorers.py) -- NOT a general n-gram ban (that stays banned by hard
    rule, no_repeat_ngram=0). This only ever bans a specific, previously-observed
    identifier token sequence on a targeted retry, never applied on a first pass.
    """
    if not banned_ngrams:
        return log_probs_t
    masked = None
    for seq in banned_ngrams:
        need = len(seq) - 1
        if need == 0:
            if masked is None:
                masked = log_probs_t.clone()
            masked[seq[0]] = float("-inf")
        elif len(generated_tail) >= need and tuple(generated_tail[-need:]) == tuple(seq[:-1]):
            if masked is None:
                masked = log_probs_t.clone()
            masked[seq[-1]] = float("-inf")
    return masked if masked is not None else log_probs_t


@torch.no_grad()
def greedy_decode(model, enc_ids, task_id, device, max_new_tokens=MAX_NEW_TOKENS,
                   collect_conf=True, collect_pgen=False, collect_attn=False,
                   stop_on_repeat_hunk=True, banned_ngrams=None):
    """KV-cached greedy decode for a single sample (batch size 1).

    banned_ngrams: optional iterable of token-id tuples. If a candidate top-1
    token would complete one of these exact sequences at the current position,
    its logit is masked to -inf (falls back to the next-best token instead).
    None/empty by default -- only ever populated by a targeted repair retry
    (see decode_rename_without_collision), never a general n-gram ban.

    Encoder runs once; the decoder only ever projects the newest token each step
    and reuses cached self/cross-attention K/V (see model_components/decoder.py's
    make_decoder_cache) -- O(max_new_tokens) decoder cost instead of the
    O(max_new_tokens^2) of re-running the full growing sequence every step.
    Mathematically identical output to feeding the whole sequence-so-far through
    model.forward() at every step and reading the last position.

    stop_on_repeat_hunk: if True (default), guards against two decode-time
    looping failure modes:
      1. Whole-hunk repeat: stop as soon as the model starts emitting a
         `<search>...<replace>...` hunk that is token-for-token identical to
         one already emitted earlier in this sample (extract_method's
         5-6x-repeated-attention-block loop; rename's occasional identical
         SEARCH/REPLACE pair emitted twice instead of covering a distinct site).
      2. Intra-hunk line repeat: stop once, within a single in-progress hunk,
         a window of 1, 2, or 3 completed lines has recurred
         MIN_REPEAT_OCCURRENCES (3) times consecutively (deduplicate's
         dominant failure mode in practice -- e.g. `def find_neighbors(...):
         return ...` emitted 3x+ in a row inside one REPLACE block, sometimes
         eating the whole max_new_tokens budget). Requiring 3 occurrences
         rather than 2 (see MIN_REPEAT_OCCURRENCES module docstring) gives
         room for benign transient duplication -- e.g. simplify collapsing an
         if/elif/else chain sometimes repeats a 1-2 line chunk once while
         "unrolling" before converging on the right answer, which a
         2-occurrence trigger was killing as a false loop. Skips windows that
         are entirely blank lines, since consecutive blank lines are
         legitimate formatting, not a loop. For the k=1 (single-line) window
         specifically, also skips lines shorter than MIN_REPEAT_LINE_TOKENS:
         short structural lines like `else:` or `pass` legitimately recur
         across separate, unrelated branches/blocks in real code.
    Neither check interacts with `no_repeat_ngram` (still 0, hard rule) -- both
    compare whole completed hunks/lines, never partial token n-grams, so
    legitimate short repeated code fragments (`self.`, indentation, etc) inside
    a single line are never suppressed.
    On trigger, decoding just stops (no synthetic <eos> is appended, and any
    already-emitted repeated content is rolled back) -- same shape as hitting
    `max_new_tokens`, already handled downstream as a truncated/non-terminated
    output.

    Returns dict with:
      'tokens'       list[int], leading <bos>, trailing <eos> if one was generated
      'mean_conf'    (if collect_conf) mean top-1 probability across steps
      'mean_entropy' (if collect_conf) mean predictive entropy (nats) across steps
      'confs'        (if collect_conf) list[float], top-1 probability per generated step
      'entropies'    (if collect_conf) list[float], predictive entropy per generated step
      'pgens'        (if collect_pgen) list[float], p_gen per generated step
      'ca_w'         (if collect_attn) np.ndarray [steps, enc_len], cross-attn per step
    """
    model.eval()
    bos_id = model.config.special_token_ids["bos"]
    eos_id = model.config.special_token_ids["eos"]
    search_id = model.config.special_token_ids["search"]
    newline_id = model.config.special_token_ids["newline"]
    LINE_REPEAT_WINDOWS = (1, 2, 3)
    MIN_REPEAT_LINE_TOKENS = 3  # k=1 only -- see docstring

    enc_t = torch.tensor(enc_ids, dtype=torch.long, device=device).unsqueeze(0)
    enc_attn = torch.ones_like(enc_t)
    task_t = torch.tensor([task_id], dtype=torch.long, device=device)

    enc_out, enc_mask = model.encode(enc_t, enc_attn=enc_attn, task_ids=task_t)
    cache = model.new_cache()

    tokens = [bos_id]
    confs, ents, pgens, attn_steps = [], [], [], []
    cur = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    seen_hunks = set()
    hunk_tokens = []
    hunk_lines = []   # completed lines (as tuples) within the current hunk
    cur_line = []     # tokens accumulated since the last newline, current hunk
    stop_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        out = model.decode_step(cur, enc_out, enc_t, cache, position_offset=step,
                                 enc_mask=enc_mask, use_copy=True)
        log_probs_t = out["log_probs"][0, -1]
        if banned_ngrams:
            log_probs_t = _mask_banned_ngrams(log_probs_t, tokens[1:], banned_ngrams)
        nxt = int(log_probs_t.argmax())

        if stop_on_repeat_hunk and nxt == search_id and hunk_tokens:
            if tuple(hunk_tokens) in seen_hunks:
                # the just-completed hunk is itself a duplicate -- roll it back
                # entirely and stop, instead of only blocking the *next* one.
                n_remove = len(hunk_tokens)
                del tokens[-n_remove:]
                if collect_conf:
                    del confs[-n_remove:]; del ents[-n_remove:]
                if collect_pgen:
                    del pgens[-n_remove:]
                if collect_attn:
                    del attn_steps[-n_remove:]
                stop_reason = "repeat_hunk"
                break
            seen_hunks.add(tuple(hunk_tokens))
            hunk_tokens, hunk_lines, cur_line = [nxt], [], []
        elif nxt == search_id:
            hunk_tokens, hunk_lines, cur_line = [nxt], [], []
        elif hunk_tokens and nxt != eos_id:
            hunk_tokens.append(nxt)
            cur_line.append(nxt)

        if collect_conf:
            probs_t = log_probs_t.exp()
            confs.append(float(probs_t.max()))
            # banned tokens (see _mask_banned_ngrams) carry log_probs_t = -inf ->
            # probs_t = 0, and 0 * -inf = nan in IEEE float, poisoning the sum.
            # Mathematically the entropy contribution of a zero-probability
            # outcome is 0, so mask those terms out explicitly instead of
            # relying on the product to vanish on its own.
            ent_terms = probs_t * log_probs_t
            ent_terms = torch.where(probs_t > 0, ent_terms, torch.zeros_like(ent_terms))
            ents.append(float(-ent_terms.sum()))
        if collect_pgen:
            pgens.append(float(out["p_gen"][0, -1]))
        if collect_attn:
            attn_steps.append(out["ca_w"][0, -1].cpu().numpy())
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long, device=device)

        if stop_on_repeat_hunk and hunk_tokens and nxt not in (newline_id, search_id, eos_id):
            token_repeat_hit = False
            for k in TOKEN_REPEAT_WINDOWS:
                n_occ = MIN_TOKEN_REPEAT_OCCURRENCES[k]
                if _tokens_repeat_n_times(hunk_tokens, k, n_occ):
                    n_remove = k * (n_occ - 1)  # keep the first cycle, drop the rest
                    del tokens[-n_remove:]
                    if collect_conf:
                        del confs[-n_remove:]; del ents[-n_remove:]
                    if collect_pgen:
                        del pgens[-n_remove:]
                    if collect_attn:
                        del attn_steps[-n_remove:]
                    stop_reason = f"repeat_token_k{k}"
                    token_repeat_hit = True
                    break
            if token_repeat_hit:
                break

        if stop_on_repeat_hunk and hunk_tokens and nxt == newline_id:
            hunk_lines.append(tuple(cur_line[:-1]))  # drop the newline itself from stored content
            cur_line = []
            for k in LINE_REPEAT_WINDOWS:
                if _lines_repeat_n_times(hunk_lines, k, MIN_REPEAT_OCCURRENCES):
                    if k == 1 and len(hunk_lines[-1]) < MIN_REPEAT_LINE_TOKENS:
                        continue
                    n_remove = sum(len(l) + 1 for l in hunk_lines[-k:])
                    del tokens[-n_remove:]
                    if collect_conf:
                        del confs[-n_remove:]; del ents[-n_remove:]
                    if collect_pgen:
                        del pgens[-n_remove:]
                    if collect_attn:
                        del attn_steps[-n_remove:]
                    stop_reason = f"repeat_line_k{k}"
                    nxt = None  # sentinel: don't also check nxt == eos_id below
                    break
            if nxt is None:
                break

        if nxt == eos_id:
            if stop_on_repeat_hunk and hunk_tokens and tuple(hunk_tokens) in seen_hunks:
                # the final hunk (immediately before this natural <eos>) is itself
                # a duplicate of an earlier one -- roll it back too, same as the
                # mid-sequence case, instead of accepting a trailing repeat.
                n_remove = len(hunk_tokens) + 1  # + this <eos>
                del tokens[-n_remove:]
                if collect_conf:
                    del confs[-n_remove:]; del ents[-n_remove:]
                if collect_pgen:
                    del pgens[-n_remove:]
                if collect_attn:
                    del attn_steps[-n_remove:]
                stop_reason = "repeat_hunk_at_eos"
            else:
                stop_reason = "eos"
            break

    result = {"tokens": tokens, "stop_reason": stop_reason}
    if collect_conf:
        result["mean_conf"] = sum(confs) / max(len(confs), 1)
        result["mean_entropy"] = sum(ents) / max(len(ents), 1)
        result["confs"] = confs
        result["entropies"] = ents
    if collect_pgen:
        result["pgens"] = pgens
    if collect_attn:
        import numpy as np
        result["ca_w"] = (np.stack(attn_steps) if attn_steps
                           else np.zeros((0, enc_t.size(1))))
    return result


@torch.no_grad()
def sample_decode(model, enc_ids, task_id, device, max_new_tokens=MAX_NEW_TOKENS,
                   temperature=1.0, stop_on_repeat_hunk=True):
    """RL-only -- not used by anything in thesis scope (RL dropped, may return as
    an appendix later). Every reported thesis number comes from greedy_decode.

    KV-cached stochastic decode for a single sample (batch size 1) -- same
    shape/guards as greedy_decode, but samples from the softmax instead of
    taking argmax at each step. Built for RL rollouts (self-critical sequence
    training, src/training/rl_training.py): needs an *exploratory* sample
    distinct from the greedy baseline so the policy-gradient advantage
    (reward(sample) - reward(greedy)) isn't trivially zero every step.

    Runs under no_grad -- the sampled tokens are replayed through a single
    parallel teacher-forced forward pass afterwards (see
    rl_training.sample_and_score) to get differentiable log-probs cheaply,
    rather than backpropagating through this autoregressive loop directly.

    Returns {'tokens': list[int]} -- same leading <bos>/trailing <eos>
    convention as greedy_decode, no confidence/pgen/attention collection
    (not needed for RL; use greedy_decode if you need those).
    """
    model.eval()
    bos_id = model.config.special_token_ids["bos"]
    eos_id = model.config.special_token_ids["eos"]
    search_id = model.config.special_token_ids["search"]
    newline_id = model.config.special_token_ids["newline"]
    LINE_REPEAT_WINDOWS = (1, 2, 3)
    MIN_REPEAT_LINE_TOKENS = 3  # k=1 only -- see greedy_decode's docstring

    enc_t = torch.tensor(enc_ids, dtype=torch.long, device=device).unsqueeze(0)
    enc_attn = torch.ones_like(enc_t)
    task_t = torch.tensor([task_id], dtype=torch.long, device=device)

    enc_out, enc_mask = model.encode(enc_t, enc_attn=enc_attn, task_ids=task_t)
    cache = model.new_cache()

    tokens = [bos_id]
    cur = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    seen_hunks = set()
    hunk_tokens = []
    hunk_lines = []
    cur_line = []
    for step in range(max_new_tokens):
        out = model.decode_step(cur, enc_out, enc_t, cache, position_offset=step,
                                 enc_mask=enc_mask, use_copy=True)
        log_probs_t = out["log_probs"][0, -1]
        probs_t = torch.softmax(log_probs_t / temperature, dim=-1)
        nxt = int(torch.multinomial(probs_t, num_samples=1))

        if stop_on_repeat_hunk and nxt == search_id and hunk_tokens:
            if tuple(hunk_tokens) in seen_hunks:
                n_remove = len(hunk_tokens)
                del tokens[-n_remove:]
                break
            seen_hunks.add(tuple(hunk_tokens))
            hunk_tokens, hunk_lines, cur_line = [nxt], [], []
        elif nxt == search_id:
            hunk_tokens, hunk_lines, cur_line = [nxt], [], []
        elif hunk_tokens and nxt != eos_id:
            hunk_tokens.append(nxt)
            cur_line.append(nxt)

        tokens.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long, device=device)

        if stop_on_repeat_hunk and hunk_tokens and nxt not in (newline_id, search_id, eos_id):
            token_repeat_hit = False
            for k in TOKEN_REPEAT_WINDOWS:
                n_occ = MIN_TOKEN_REPEAT_OCCURRENCES[k]
                if _tokens_repeat_n_times(hunk_tokens, k, n_occ):
                    n_remove = k * (n_occ - 1)
                    del tokens[-n_remove:]
                    token_repeat_hit = True
                    break
            if token_repeat_hit:
                break

        if stop_on_repeat_hunk and hunk_tokens and nxt == newline_id:
            hunk_lines.append(tuple(cur_line[:-1]))
            cur_line = []
            stop_here = False
            for k in LINE_REPEAT_WINDOWS:
                if _lines_repeat_n_times(hunk_lines, k, MIN_REPEAT_OCCURRENCES):
                    if k == 1 and len(hunk_lines[-1]) < MIN_REPEAT_LINE_TOKENS:
                        continue
                    n_remove = sum(len(l) + 1 for l in hunk_lines[-k:])
                    del tokens[-n_remove:]
                    stop_here = True
                    break
            if stop_here:
                break

        if nxt == eos_id:
            if stop_on_repeat_hunk and hunk_tokens and tuple(hunk_tokens) in seen_hunks:
                n_remove = len(hunk_tokens) + 1
                del tokens[-n_remove:]
            break

    return {"tokens": tokens}


def decode_rename_without_collision(model, enc_ids, task_id, device, before_ids, tokenizer, config,
                                     max_retries=2, **greedy_kwargs):
    """Greedy-decode a rename sample, then detect + repair the name-collision
    failure mode (two distinct before-identifiers renamed to the identical new
    name -- e.g. self.u()/self.d() both -> update_status -- see
    project_rename_failure_breakdown, idx 55/68/88) via a targeted
    re-decode that bans the specific colliding name's token sequence.

    Costs nothing extra beyond a plain greedy_decode when no collision is
    detected (the common case). Only re-decodes -- up to `max_retries` times,
    accumulating bans across retries -- when find_pred_collisions actually
    finds one. Does not touch apply_sr failures (parse_fail) or any other
    failure mode; those are a different problem, left alone here.

    Known limitation: a colliding name's token-id sequence is obtained by
    encoding it standalone (tokenizer.encode(name)), which can differ from its
    in-context BPE boundary (e.g. right after `def ` vs after `self.`) -- the
    ban may occasionally miss a collision whose surrounding context tokenizes
    the name differently. Accepted as a heuristic, not a guarantee.

    Returns the same dict shape as greedy_decode, plus:
      'collision_retries'    int, how many repair passes were actually run
      'collisions_remaining' set of new-name strings still colliding after
                              max_retries (empty if resolved or never present)
    """
    from src.eval.scorers import find_pred_collisions
    from src.data_set import apply_sr

    before_text = tokenizer.decode(before_ids)
    banned_ngrams = set()
    out = greedy_decode(model, enc_ids, task_id, device, banned_ngrams=banned_ngrams, **greedy_kwargs)

    n_retries = 0
    collisions = set()
    while True:
        pred_after = apply_sr(before_ids, out["tokens"], config)
        collisions = (find_pred_collisions(before_text, tokenizer.decode(pred_after))
                      if pred_after is not None else set())
        if not collisions or n_retries >= max_retries:
            break
        newline_id = config.special_token_ids["newline"]
        for name in collisions:
            # tokenizer.encode() runs a bare identifier through the same
            # code-oriented pipeline as a full snippet, which appends a
            # trailing <newline> that is not part of the identifier itself
            # (verified empirically: encode("update_status") ->
            # [update_, status</w>, <newline>]) -- keeping it would ban
            # "name followed by newline" instead of the name itself, which
            # never actually completes the collision (the real next token
            # after the name's first sub-token is whatever code follows it
            # in context, never a newline). Strip it before banning.
            name_ids = tuple(t for t in tokenizer.encode(name) if t != newline_id)
            if name_ids:
                banned_ngrams.add(name_ids)
        n_retries += 1
        out = greedy_decode(model, enc_ids, task_id, device, banned_ngrams=banned_ngrams, **greedy_kwargs)

    out["collision_retries"] = n_retries
    out["collisions_remaining"] = collisions
    return out
