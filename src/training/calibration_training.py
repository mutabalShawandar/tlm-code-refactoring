"""Per-token, correctness-supervised confidence calibration on the model's own
greedy free-decode rollouts.

Root cause being targeted: teacher forcing only ever trains the model on
correct prefixes, so token-level confidence tracks self-consistency with a
gold-only history, not actual correctness. At free-decode inference the model
conditions on its own (sometimes wrong) history -- a state never seen in
training -- and stays confident anyway. This module closes that gap directly:
it rolls the model out greedily on its own trajectory (no_grad, same
KV-cached path as inference), labels each emitted token 1/0 against the gold
target via token-id alignment, then replays that trajectory through one
ordinary teacher-forced forward pass to get differentiable per-token
confidences and trains those against the correctness labels (Brier score).

The gold teacher-forced loss (compute_loss, which already includes the
role-aware p_gen anchor) is mixed in as an anchor on every step -- same
"never let a distribution-reshaping term run unanchored" lesson RL's
MLE-mixed loss learned the hard way (rl_training.py) -- so the calibration
term cannot flatten generation on its own.

Batching (added after profiling showed ~18s/step at MINIBATCH=16 on Colab
GPU, dominated by running the whole minibatch at batch size 1 in a Python
loop -- see calibration_step's docstring for the breakdown):
  - the encoder now runs exactly TWICE per minibatch -- once no_grad (for the
    rollout) and once grad-enabled, shared by BOTH the rollout-replay and the
    gold-anchor pass via `TLM_S.decode_full` (model_variants/tlm_s.py) -- down
    from up to 3 encodes PER RECORD in the old per-record loop (1 inside
    greedy_decode + 1 inside each of the two `forward()` calls, which always
    re-encode). The no_grad rollout's `enc_out` can't be reused for the
    differentiable passes -- it carries no gradient history -- so two encodes
    total is the real floor here, not one;
  - the rollout itself is generated for the whole minibatch in lockstep
    (`batched_greedy_rollout`) instead of one record at a time, so each decode
    step is one batched GPU forward instead of B sequential ones;
  - the rollout-replay and gold-anchor forward passes are each a single
    padded batched call instead of a per-record loop.

`batched_greedy_rollout` is a deliberately simplified, training-only
generation path -- see its docstring for exactly what it drops relative to
`src/inference/generate.py::greedy_decode` (the shared eval/inference path,
left untouched) and why that's acceptable here.
"""

from difflib import SequenceMatcher

import torch

from src.training.training import sr_role_masks, compute_loss

SRC_VERSION = "calibration-selfrollout-v2-batched"
print(f"[calibration_training.py] SRC_VERSION = {SRC_VERSION}")


def token_correctness_labels(dec_tgt_list, gold_tgt_list):
    """Per-token 1/0 correctness for each position of the model's own emitted
    target stream (`pred[1:]`), aligned to the gold target stream
    (`gold[1:]`) via token-id `SequenceMatcher`.

    Structural tokens (<search>/<replace>/<eos>) are left in the alignment --
    harmless, since `compute_calibration_loss` masks them out via
    `content_mask` regardless of their label here.

    Returns a `list[int]` the same length as `dec_tgt_list`.
    """
    labels = [0] * len(dec_tgt_list)
    sm = SequenceMatcher(None, dec_tgt_list, gold_tgt_list, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                labels[i] = 1
    return labels


def compute_calibration_loss(log_probs, dec_tgt, correctness, pad_id, use_brier=True):
    """Brier (default) or BCE calibration loss between the model's own
    confidence in the token it emitted and whether that token was actually
    correct, restricted to content tokens. Fully batch-general: (B, T, V) /
    (B, T) / (B, T) all work as-is, aggregated over the WHOLE batch (not
    averaged per-record first) -- same aggregation convention `compute_loss`
    already uses for the gold anchor.

    log_probs:   (B, T, V) from a teacher-forced replay of the ROLLOUT.
    dec_tgt:     (B, T)    the rollout's own emitted tokens (`pred[1:]`),
                 padded with `pad_id` across the batch.
    correctness: (B, T)    float 0/1 labels aligned to `dec_tgt` (padding
                 position values are irrelevant -- masked out below).

    Confidence = probability the model assigned to the token it actually
    emitted. Restricted to content tokens via `sr_role_masks`'s
    `content_mask` (excludes <search>/<replace>/<eos>/PAD). Brier (squared
    error) by default -- numerically safe; BCE needs clamping and can blow up
    as conf -> 1 with y = 0, the common early-training case.
    """
    conf = log_probs.gather(-1, dec_tgt.unsqueeze(-1)).squeeze(-1).exp()   # (B, T)
    _, _, _, content_mask = sr_role_masks(dec_tgt, pad_id)                 # (B, T)
    denom = content_mask.sum().clamp(min=1.0)
    if use_brier:
        per_tok = (conf - correctness) ** 2
    else:
        c = conf.clamp(1e-6, 1 - 1e-6)
        per_tok = -(correctness * torch.log(c) + (1 - correctness) * torch.log(1 - c))
    return (per_tok * content_mask).sum() / denom


def _pad_batch(seqs, pad_id, device):
    """list[list[int]] -> (B, T) LongTensor, right-padded with pad_id to the
    batch's own max length."""
    max_len = max(len(s) for s in seqs)
    padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _amp_ctx(device, use_amp):
    """torch.autocast on CUDA when requested, a no-op context otherwise (CPU
    autocast for this model isn't worth the numerical fuss -- there's no
    Colab-GPU-underutilization problem to solve on CPU in the first place)."""
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def batched_greedy_rollout(model, enc_ids_list, task_ids_list, device, pad_id,
                            max_new_tokens=512, use_amp=True):
    """Greedy-decode a whole minibatch in lockstep (one batched forward per
    decode step instead of one sequential rollout per record) -- the training
    counterpart of `src/inference/generate.py::greedy_decode`, NOT a
    replacement for it (that function is untouched and still used everywhere
    for eval/inference).

    Deliberately drops two things `greedy_decode` has, both scoped to
    generation *quality* rather than correctness of the calibration signal:
      - `stop_on_repeat_hunk`'s whole-hunk/line/token repeat guards. Those
        guards are per-record stateful bookkeeping that doesn't batch cleanly,
        and skipping them here is harmless for THIS use: an occasional
        repeated hunk in a training rollout still gets faithfully labeled
        wrong (or right) by `token_correctness_labels` against gold, which is
        legitimate training signal either way -- it just means a rollout can
        occasionally run longer before its own natural EOS than it would
        under the guarded eval path.
      - confidence/entropy/p_gen collection (`collect_conf` etc.) -- unneeded
        here since only the token ids feed the correctness labels.

    Each record stops independently at its own first emitted <eos> (tracked
    per-row); the whole batch stops once every row has finished or
    `max_new_tokens` is hit. Rows that finish early keep getting stepped
    (their output is simply discarded past their own EOS) since the batched
    KV cache has no per-row early-exit -- harmless, batch rows don't interact.

    Returns list[list[int]], one token-id list per record (`[bos, ..., eos?]`,
    same convention as `greedy_decode`: trailing eos only if one was actually
    generated within budget).
    """
    bos_id = model.config.special_token_ids["bos"]
    eos_id = model.config.special_token_ids["eos"]
    B = len(enc_ids_list)

    enc_t = _pad_batch(enc_ids_list, pad_id, device)
    enc_attn = (enc_t != pad_id).long()
    task_t = torch.tensor(task_ids_list, dtype=torch.long, device=device)

    model.eval()
    with _amp_ctx(device, use_amp):
        enc_out, enc_mask = model.encode(enc_t, enc_attn=enc_attn, task_ids=task_t)
    cache = model.new_cache()

    tokens = [[bos_id] for _ in range(B)]
    finished = [False] * B
    cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        with _amp_ctx(device, use_amp):
            out = model.decode_step(cur, enc_out, enc_t, cache, position_offset=step,
                                     enc_mask=enc_mask, use_copy=True)
        nxt = out["log_probs"][:, -1, :].float().argmax(dim=-1)   # (B,)
        nxt_list = nxt.tolist()
        for b in range(B):
            if finished[b]:
                continue
            tokens[b].append(nxt_list[b])
            if nxt_list[b] == eos_id:
                finished[b] = True
        if all(finished):
            break
        cur = nxt.unsqueeze(1)

    return tokens


def calibration_step(model, records, tokenizer, config, device, pad_id,
                      make_sft_sr, lambda_cal=1.0, loss_fn=compute_loss,
                      max_new_tokens=512, use_amp=True):
    """One minibatch, batched end-to-end (see module docstring for why this
    replaced an earlier per-record-loop version): one batched no_grad rollout
    (its own encode), then one grad-enabled encode shared by two batched
    decode_full passes (rollout-replay + gold-anchor).

    L = L_task_gold  +  lambda_cal * L_cal_selfrollout

    Both terms are batch-aggregated (mean per content token across the whole
    minibatch, not a per-record mean of per-record means) -- the same
    aggregation convention `compute_loss`/`_step_batch` already use elsewhere,
    and a cleaner fit here than the RL template's per-record loop since
    (unlike RL) there is no per-record reward/advantage weighting to
    preserve.

    `use_amp=True` runs the encode/decode forward passes under
    `torch.autocast` on CUDA (no-op on CPU). Loss values are always computed
    in float32 (log_probs cast back before compute_calibration_loss/loss_fn)
    regardless of autocast, since Brier/NLL involve log/exp terms that are
    worth keeping numerically exact. Pair with `torch.cuda.amp.GradScaler` in
    the caller's backward/optimizer step for correct fp16 training (see the
    calibration notebook's Cell 7) -- autocast alone only saves compute, it
    doesn't handle fp16 gradient underflow by itself.

    Returns None if every record in the minibatch was unusable (encode
    failure or degenerate rollout), else a dict with 'loss' (differentiable),
    'mean_cal', 'mean_task' (both floats, for logging), and 'n'.
    """
    parsed = []
    for d in records:
        res = make_sft_sr(d, tokenizer, config)
        if res is None:
            continue
        enc_ids, gold_dec_ids, task_id = res
        parsed.append((enc_ids, gold_dec_ids, task_id))
    if not parsed:
        return None

    enc_ids_list = [p[0] for p in parsed]
    task_ids_list = [p[2] for p in parsed]

    # --- 1. rollout the whole minibatch on the model's OWN greedy trajectory (no grad) ---
    rollouts = batched_greedy_rollout(model, enc_ids_list, task_ids_list, device, pad_id,
                                       max_new_tokens=max_new_tokens, use_amp=use_amp)
    # Defensive, not just a caller convention: batched_greedy_rollout forces
    # model.eval() internally (needed for a clean no-dropout rollout), which
    # would otherwise silently disable dropout on the replay/anchor passes
    # below -- same bug class rl_training.py's self_critical_step guards
    # against for the same reason (see its comment on this exact line).
    model.train()

    keep = [i for i, r in enumerate(rollouts) if len(r) >= 3]   # need >=1 content token
    if not keep:
        return None

    enc_ids_kept = [enc_ids_list[i] for i in keep]
    task_ids_kept = [task_ids_list[i] for i in keep]
    gold_kept = [parsed[i][1] for i in keep]
    pred_kept = [rollouts[i] for i in keep]
    n = len(keep)

    # --- 2. one shared encode for both passes below ---
    enc_t = _pad_batch(enc_ids_kept, pad_id, device)
    enc_attn = (enc_t != pad_id).long()
    task_t = torch.tensor(task_ids_kept, dtype=torch.long, device=device)
    with _amp_ctx(device, use_amp):
        enc_out, enc_mask = model.encode(enc_t, enc_attn=enc_attn, task_ids=task_t)

    # --- 3. rollout-replay pass -> calibration loss ---
    pred_tgt_kept = [p[1:] for p in pred_kept]
    y_kept = [token_correctness_labels(pred_tgt_kept[j], gold_kept[j][1:]) for j in range(n)]
    dec_in = _pad_batch([p[:-1] for p in pred_kept], pad_id, device)
    dec_tgt = _pad_batch(pred_tgt_kept, pad_id, device)
    y_t = _pad_batch(y_kept, 0, device).float()
    with _amp_ctx(device, use_amp):
        out = model.decode_full(enc_out, enc_mask, enc_t, dec_in,
                                 dec_attn=(dec_in != pad_id).long(), use_copy=True)
    L_cal = compute_calibration_loss(out["log_probs"].float(), dec_tgt, y_t, pad_id)

    # --- 4. gold teacher-forced anchor pass (standard loss, differentiable) ---
    g_in = _pad_batch([g[:-1] for g in gold_kept], pad_id, device)
    g_tgt = _pad_batch([g[1:] for g in gold_kept], pad_id, device)
    with _amp_ctx(device, use_amp):
        g_out = model.decode_full(enc_out, enc_mask, enc_t, g_in,
                                   dec_attn=(g_in != pad_id).long(), use_copy=True)
    g_out = {**g_out, "log_probs": g_out["log_probs"].float(),
             "p_gen": g_out["p_gen"].float() if g_out.get("p_gen") is not None else None}
    L_task, _, _ = loss_fn(g_out, g_tgt, pad_id, task_ids=task_t)

    loss = L_task + lambda_cal * L_cal
    return {"loss": loss, "mean_cal": float(L_cal.detach()), "mean_task": float(L_task.detach()), "n": n}
