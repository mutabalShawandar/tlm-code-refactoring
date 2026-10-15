import math
import torch
import torch.nn.functional as F

# -- src version -- printed on import so training logs confirm the right code -
SRC_VERSION = "role-aware-pgen-v4-intuitive-last"
print(f"[training.py] SRC_VERSION = {SRC_VERSION}")

# -- role-aware p_gen targets (SR format) ----------------------------------
# Token IDs from the project spec (post bugs-11/12/13 vocab): 14=search, 15=replace
# Fixed later: these were hardcoded to 17/18 (the pre-fix vocab's ids, which now
# belong to extra_id_1/extra_id_2), silently misclassifying SEARCH/REPLACE spans for
# the entire epoch-20 fixed-vocab pretrain run.
#
# NOTE on polarity: p_gen high=generate/low=copy is the correct convention
# (see CopyPointerHead: P_final = p_gen*P_vocab + (1-p_gen)*P_copy). It is
# tempting to therefore set SEARCH (should be copied) low and REPLACE
# (often new content) high -- that "intuitive" polarity was tried twice
# (v2: flat 0.8 REPLACE; v3: task-aware 0.3/0.8 REPLACE) and both collapsed
# rename to ~0-0.7% applied_ok. The ORIGINAL v1 values (which look backwards
# relative to that intuition) are the only ones with a confirmed healthy
# result (56% rename, kernel version 8). v4 (current SR values below) is one
# more from-scratch attempt at the intuitive polarity. compute_loss uses these
# values unchanged -- see compute_loss_free_pgen below for the separate,
# non-role-conditioned alternative ("let the model decide").
_SEARCH_ID   = 14
_REPLACE_ID  = 15
_EOS_ID      = 3     # excluded from the penalty like the structural markers are
PGEN_SEARCH  = 0.2   # SEARCH block target
PGEN_REPLACE = 0.8   # REPLACE block target
PGEN_NEUTRAL = 0.45  # fallback for structural tokens / full-after format
PGEN_LAMBDA  = 1.0   # weight on role-aware MSE penalty

# -- p_gen: anti-collapse floor only, for compute_loss_free_pgen -----------
# Project spec hard rule: one-sided floor, only fires below PGEN_FLOOR, no upward
# pressure once p_gen clears it -- doesn't fight the NLL gradient once p_gen is
# in a healthy range. Historically the ONLY p_gen mechanism validated at this
# lambda (p_gen stabilized ~0.7 during full training, no instability) -- from
# before the role-aware MSE penalty above existed at all.
# Not used by compute_loss (unchanged, still role-aware) -- only by
# compute_loss_free_pgen, the "let the model decide" experiment.
PGEN_FLOOR        = 0.4
PGEN_FLOOR_LAMBDA = 0.1

# -- coverage penalty (See et al. 2017 pointer-generator) ------------------
# Separate mechanism from p_gen above -- targets cross-attention *position
# tracking*, not the copy/generate balance. Added after
# the SFT eval notebook's cross-attention plots showed the worst
# rename/deduplicate/extract_method samples losing track of where they are
# in the source: attention jumping backward to re-copy an already-emitted
# span, or orbiting the same early encoder region across every SR hunk
# instead of advancing. Cross-attention has no positional signal at all
# (RoPE is self-attention only), so it's purely content-addressed -- nothing
# stops it re-selecting a position it already drew from. This penalizes
# exactly that. Starting value, NOT yet empirically validated -- A/B against
# a COVERAGE_LAMBDA=0 baseline before trusting any result (this project has
# been burned twice by trusting a hyperparameter choice on theory alone).
COVERAGE_LAMBDA = 1.0

############ Never Tried Just a nice Idea #################### 
# FA: per-task targets (task_id -> p_gen target)
# rename/simplify/dedup/remove_dead: output is mostly before -> lean copy (low p_gen)
# extract_method: output contains new helper function -> lean generate (high p_gen)
# PGEN_FA_BY_TASK = {
#     0: 0.45,  # pretrain -- neutral fallback
#     1: 0.40,  # rename
#     2: 0.40,  # simplify
#     3: 0.40,  # deduplicate
#     4: 0.40,  # remove_dead_code
#     5: 0.70,  # extract_method
# }


def sr_role_masks(dec_target, pad_id: int):
    """Per-token SR role classification, extracted from `compute_loss` so it can
    be exercised directly by a round-trip check (see
    `src/utils/pgen_role_check.py`) against an
    independently-computed ground truth, rather than only being verifiable by
    reading the vectorized cumsum logic.

    Returns (in_search, in_replace, is_structural, content_mask), each a
    (B, T) bool/float tensor aligned with `dec_target`. `is_structural` covers
    <search>/<replace>/<eos>; PAD is excluded via `content_mask` (multiplied
    with the caller's own non-pad mask), not via role classification, since a
    pad position's role label is irrelevant once masked out."""
    device = dec_target.device
    B = dec_target.size(0)
    mask = (dec_target != pad_id).float()

    search_tok  = (dec_target == _SEARCH_ID).float()
    replace_tok = (dec_target == _REPLACE_ID).float()
    cum_s = torch.cumsum(search_tok,  dim=1)
    cum_r = torch.cumsum(replace_tok, dim=1)
    prev_s = torch.cat([torch.zeros(B, 1, device=device), cum_s[:, :-1]], dim=1)
    prev_r = torch.cat([torch.zeros(B, 1, device=device), cum_r[:, :-1]], dim=1)

    is_structural = (dec_target == _SEARCH_ID) | (dec_target == _REPLACE_ID) | (dec_target == _EOS_ID)
    in_search  = (prev_s > prev_r)  & ~is_structural
    in_replace = (prev_s == prev_r) & (prev_s > 0) & ~is_structural

    content_mask = mask * (~is_structural).float()
    return in_search, in_replace, is_structural, content_mask


def compute_loss(out, dec_target, pad_id: int, task_ids=None):
    """
    Teacher-forced NLL + role-aware p_gen MSE penalty.

    SR format (decoder contains <search>/<replace> tokens):
      Flat per-role target: SEARCH block -> PGEN_SEARCH, REPLACE block -> PGEN_REPLACE
      (see the note by those constants -- do not re-derive from "high=generate,
      low=copy" theory without re-validating against the v1 baseline).
      Structural tokens, <eos>, and PAD excluded from penalty.

    FA format (no SR markers detected):
      Per-task flat target from PGEN_FA_BY_TASK (task_ids required).
      rename/simplify/dedup/remove_dead_code -> 0.4 (lean copy)
      extract_method -> 0.7 (lean generate)

    Returns (loss, nll, p_gen_mean).
    """
    log_probs = out["log_probs"]                       # (B, T, V)
    V = log_probs.size(-1)
    nll = F.nll_loss(
        log_probs.reshape(-1, V),
        dec_target.reshape(-1),
        ignore_index=pad_id,
        reduction="mean",
    )
    loss = nll
    pgen_mean = torch.tensor(0.0, device=log_probs.device)

    p_gen = out.get("p_gen")
    if p_gen is not None and PGEN_LAMBDA > 0:
        device = dec_target.device
        B = dec_target.size(0)
        mask = (dec_target != pad_id).float()          # (B, T)
        denom = mask.sum().clamp(min=1.0)
        pgen_mean = (p_gen * mask).sum() / denom

        has_sr = ((dec_target == _SEARCH_ID) | (dec_target == _REPLACE_ID)).any()

        if has_sr:
            # -- SR format: role-aware per-token targets (flat, v1 values) --
            # excluded from the penalty: structural markers themselves and <eos>
            # (bug fix: <eos> used to fall into in_replace since prev_s==prev_r
            # holds after the last pair closes, silently pulling it toward
            # PGEN_REPLACE)
            in_search, in_replace, is_structural, content_mask = sr_role_masks(dec_target, pad_id)

            pgen_target = torch.full_like(p_gen, PGEN_NEUTRAL)
            pgen_target[in_search]  = PGEN_SEARCH
            pgen_target[in_replace] = PGEN_REPLACE
        else:
            # -- FA format: per-task flat target --------------------------
            if task_ids is not None:
                fa_t = torch.tensor(
                    [PGEN_FA_BY_TASK.get(int(tid), PGEN_NEUTRAL) for tid in task_ids],
                    dtype=p_gen.dtype, device=device,
                )                                      # (B,)
                pgen_target = fa_t.unsqueeze(1).expand_as(p_gen)
            else:
                pgen_target = torch.full_like(p_gen, PGEN_NEUTRAL)
            content_mask = mask

        denom_c = content_mask.sum().clamp(min=1.0)
        pgen_pen = (((p_gen - pgen_target) ** 2) * content_mask).sum() / denom_c
        loss = loss + PGEN_LAMBDA * pgen_pen

        ca_w = out.get("ca_w")
        if ca_w is not None and COVERAGE_LAMBDA > 0:
            # coverage_prev[b, t, :] = sum of attention paid to each encoder
            # position at all PRIOR decode steps (exclusive cumsum along T).
            # Penalize min(current attention, prior coverage) per encoder
            # position -- high only when this step re-attends somewhere
            # already drawn from heavily, near-zero for genuinely new ground.
            coverage_prev = torch.cumsum(ca_w, dim=1) - ca_w        # (B, T, Se)
            step_cov_loss = torch.minimum(ca_w, coverage_prev).sum(dim=-1)  # (B, T)
            coverage_pen = (step_cov_loss * content_mask).sum() / denom_c
            loss = loss + COVERAGE_LAMBDA * coverage_pen

    return loss, nll.detach(), pgen_mean.detach()


# -- "let the model decide" (p_gen anti-collapse floor, no role target) ----
# Separate function rather than a flag on compute_loss, same convention as
# compute_loss_label_smoothed below, so compute_loss's SR/FA role-aware path
# (v1-v4 history) stays completely untouched.
#
# Rationale: a flat per-block target assumes SEARCH/REPLACE are internally
# homogeneous (all-copy / all-generate), but most real edits are small and
# local -- a REPLACE block is usually mostly unchanged context copied from
# SEARCH, with only a narrow new span inside it. Forcing the whole block
# toward one p_gen value, in either direction, fights copying of that
# unchanged majority. p_gen is a per-token linear gate; there's no structural
# reason it can't learn a token-conditional copy/generate balance under NLL
# gradient alone, given enough anti-collapse pressure to avoid the degenerate
# collapse role-conditioning was originally added to prevent. See
# PGEN_FLOOR/PGEN_FLOOR_LAMBDA above -- untested as of this writing, no prior
# result either way (unlike compute_loss's v2/v3, which have a confirmed
# negative result at the intuitive polarity).
def compute_loss_free_pgen(out, dec_target, pad_id: int, task_ids=None):
    """
    Teacher-forced NLL + p_gen anti-collapse floor only -- no SEARCH/REPLACE
    or per-task role target. Format-agnostic (SR and FA content get identical
    p_gen treatment); only the coverage penalty still distinguishes SR from FA,
    since that exclusion is about cross-attention position tracking, not about
    what p_gen is pushed toward.

    p_gen floor: PGEN_FLOOR_LAMBDA * relu(PGEN_FLOOR - p_gen), averaged over
    non-pad positions. Only fires below PGEN_FLOOR; free (no gradient from this
    term) everywhere above it.

    Returns (loss, nll, p_gen_mean) -- same signature as compute_loss.
    """
    log_probs = out["log_probs"]                       # (B, T, V)
    V = log_probs.size(-1)
    nll = F.nll_loss(
        log_probs.reshape(-1, V),
        dec_target.reshape(-1),
        ignore_index=pad_id,
        reduction="mean",
    )
    loss = nll
    pgen_mean = torch.tensor(0.0, device=log_probs.device)

    p_gen = out.get("p_gen")
    if p_gen is not None:
        mask = (dec_target != pad_id).float()          # (B, T)
        denom = mask.sum().clamp(min=1.0)
        pgen_mean = (p_gen * mask).sum() / denom

        if PGEN_FLOOR_LAMBDA > 0:
            floor_pen = (F.relu(PGEN_FLOOR - p_gen) * mask).sum() / denom
            loss = loss + PGEN_FLOOR_LAMBDA * floor_pen

        ca_w = out.get("ca_w")
        if ca_w is not None and COVERAGE_LAMBDA > 0:
            has_sr = ((dec_target == _SEARCH_ID) | (dec_target == _REPLACE_ID)).any()
            if has_sr:
                _, _, _, content_mask = sr_role_masks(dec_target, pad_id)
            else:
                content_mask = mask
            denom_c = content_mask.sum().clamp(min=1.0)
            coverage_prev = torch.cumsum(ca_w, dim=1) - ca_w        # (B, T, Se)
            step_cov_loss = torch.minimum(ca_w, coverage_prev).sum(dim=-1)  # (B, T)
            coverage_pen = (step_cov_loss * content_mask).sum() / denom_c
            loss = loss + COVERAGE_LAMBDA * coverage_pen

    return loss, nll.detach(), pgen_mean.detach()


# -- label smoothing (exposure-bias / free-decode-overconfidence remedy) ---
# Planned for the TLM-XS fixed-vocab pretrain retry only (see
# project_label_smoothing_pilot_planned / project_exposure_bias_ceiling_argument
# memory) -- the 4th, structurally distinct candidate remedy after RL,
# RL-conf-calib, and scheduled sampling all failed to close the free-decode
# calibration gap. Kept as a separate function rather than a flag on
# compute_loss so the S/M code path above is untouched.
#
# Implementation gotcha: `log_probs` (out["log_probs"]) is already the
# copy-pointer mixture's log-density, not a plain softmax over vocab logits,
# so PyTorch's F.cross_entropy(..., label_smoothing=) cannot be used here (it
# would smooth the wrong distribution). The smoothed target is built manually
# and compared against log_probs directly via -sum(target * log_probs),
# which is correct regardless of what produced log_probs.
def label_smoothed_nll(log_probs, dec_target, pad_id: int, smoothing: float):
    """-sum(smoothed_target * log_probs), per-token, mean over non-pad tokens.
    smoothed_target puts (1-smoothing) mass on dec_target and smoothing mass
    spread uniformly over the whole vocab (fairseq-style label smoothing)."""
    nll_per_tok = -log_probs.gather(dim=-1, index=dec_target.unsqueeze(-1)).squeeze(-1)  # (B, T)
    smooth_per_tok = -log_probs.mean(dim=-1)                                             # (B, T)
    per_tok = (1.0 - smoothing) * nll_per_tok + smoothing * smooth_per_tok
    mask = (dec_target != pad_id).float()
    denom = mask.sum().clamp(min=1.0)
    return (per_tok * mask).sum() / denom


def compute_loss_label_smoothed(out, dec_target, pad_id: int, task_ids=None, smoothing: float = 0.1):
    """Same as compute_loss (role-aware p_gen penalty + coverage penalty
    unchanged) but the teacher-forced NLL term is replaced by
    label_smoothed_nll for gradient purposes. The unsmoothed `nll` is still
    computed and returned for logging/comparison, matching compute_loss's
    return signature.

    Returns (loss, nll, p_gen_mean) -- nll is the unsmoothed diagnostic value.
    """
    log_probs = out["log_probs"]                       # (B, T, V)
    V = log_probs.size(-1)
    nll = F.nll_loss(
        log_probs.reshape(-1, V),
        dec_target.reshape(-1),
        ignore_index=pad_id,
        reduction="mean",
    )
    loss = label_smoothed_nll(log_probs, dec_target, pad_id, smoothing)
    pgen_mean = torch.tensor(0.0, device=log_probs.device)

    p_gen = out.get("p_gen")
    if p_gen is not None and PGEN_LAMBDA > 0:
        device = dec_target.device
        B = dec_target.size(0)
        mask = (dec_target != pad_id).float()          # (B, T)
        denom = mask.sum().clamp(min=1.0)
        pgen_mean = (p_gen * mask).sum() / denom

        has_sr = ((dec_target == _SEARCH_ID) | (dec_target == _REPLACE_ID)).any()

        if has_sr:
            in_search, in_replace, is_structural, content_mask = sr_role_masks(dec_target, pad_id)

            pgen_target = torch.full_like(p_gen, PGEN_NEUTRAL)
            pgen_target[in_search]  = PGEN_SEARCH
            pgen_target[in_replace] = PGEN_REPLACE
        else:
            if task_ids is not None:
                fa_t = torch.tensor(
                    [PGEN_FA_BY_TASK.get(int(tid), PGEN_NEUTRAL) for tid in task_ids],
                    dtype=p_gen.dtype, device=device,
                )                                      # (B,)
                pgen_target = fa_t.unsqueeze(1).expand_as(p_gen)
            else:
                pgen_target = torch.full_like(p_gen, PGEN_NEUTRAL)
            content_mask = mask

        denom_c = content_mask.sum().clamp(min=1.0)
        pgen_pen = (((p_gen - pgen_target) ** 2) * content_mask).sum() / denom_c
        loss = loss + PGEN_LAMBDA * pgen_pen

        ca_w = out.get("ca_w")
        if ca_w is not None and COVERAGE_LAMBDA > 0:
            coverage_prev = torch.cumsum(ca_w, dim=1) - ca_w        # (B, T, Se)
            step_cov_loss = torch.minimum(ca_w, coverage_prev).sum(dim=-1)  # (B, T)
            coverage_pen = (step_cov_loss * content_mask).sum() / denom_c
            loss = loss + COVERAGE_LAMBDA * coverage_pen

    return loss, nll.detach(), pgen_mean.detach()


def _step_batch(model, batch, device, pad_id: int, loss_fn=compute_loss):
    """loss_fn defaults to compute_loss (S/M behavior, unchanged). Pass
    compute_loss_label_smoothed (e.g. via functools.partial to bind
    `smoothing`) for the TLM-XS exposure-bias pilot."""
    enc_ids  = batch["enc_ids"].to(device)
    enc_attn = batch["enc_attn"].to(device)
    dec_ids  = batch["dec_ids"].to(device)
    task_ids = batch["task_ids"].to(device)

    dec_in  = dec_ids[:, :-1]
    dec_tgt = dec_ids[:, 1:]
    dec_attn = (dec_in != pad_id).long()

    out = model(enc_ids, dec_in, enc_attn=enc_attn, dec_attn=dec_attn,
                task_ids=task_ids, use_copy=True)
    return loss_fn(out, dec_tgt, pad_id, task_ids=task_ids)


def _step_batch_scheduled(model, batch, device, pad_id: int, sampler, p_sample):
    """
    Scheduled-sampling step (2-pass parallel approximation):
    Pass 1 gets teacher-forced argmax predictions with no grad.
    Pass 2 re-runs with mixed inputs (gold vs model prediction per token) with grad.
    Targets are unchanged -- only decoder inputs shift.
    Do NOT use with copy pointer head (parallel SS breaks copy-gate state -- see the project spec).
    """
    enc_ids  = batch["enc_ids"].to(device)
    enc_attn = batch["enc_attn"].to(device)
    dec_ids  = batch["dec_ids"].to(device)
    task_ids = batch["task_ids"].to(device)

    dec_in  = dec_ids[:, :-1]
    dec_tgt = dec_ids[:, 1:]
    dec_attn = (dec_in != pad_id).long()

    with torch.no_grad():
        out0 = model(enc_ids, dec_in, enc_attn=enc_attn, dec_attn=dec_attn,
                     task_ids=task_ids, use_copy=True)
        pred = out0["log_probs"].argmax(dim=-1)

    mixed = dec_in.clone()
    mixed[:, 1:] = sampler.sample_or_teach(pred[:, :-1], dec_in[:, 1:], p_sample, pad_id)

    out = model(enc_ids, mixed, enc_attn=enc_attn, dec_attn=dec_attn,
                task_ids=task_ids, use_copy=True)
    return compute_loss(out, dec_tgt, pad_id, task_ids=task_ids)


def _step_batch_scheduled_sequential(model, batch, device, pad_id: int, sampler, p_sample):
    """
    Sequential scheduled-sampling step -- safe to use with the copy pointer
    head, unlike _step_batch_scheduled's 2-pass parallel approximation (see
    that function's docstring / the project spec's hard rule: "no parallel scheduled
    sampling, broke copy gate, Try #3 regression").

    Builds the mixed decoder input autoregressively, one position at a time,
    via the model's KV-cached incremental decode path (model.new_cache() /
    model.decode_step() -- the same mechanism src/inference/generate.py's
    greedy_decode uses at inference), so each injected token is exactly what
    the model would have produced from the *actual* mixed-so-far trajectory,
    not a one-shot argmax computed from a fully-gold-conditioned pass. Cost is
    O(T) cached forward steps for the rollout, not O(T^2).

    The rollout is entirely no_grad (truncated backprop-through-time, same
    convention as rl_training.py's self-critical rollouts). The final loss
    comes from one ordinary teacher-forced forward pass over the resulting
    mixed sequence (same shape/cost as _step_batch) -- gradients only flow
    through that single pass, never through the rollout that produced it.
    """
    enc_ids  = batch["enc_ids"].to(device)
    enc_attn = batch["enc_attn"].to(device)
    dec_ids  = batch["dec_ids"].to(device)
    task_ids = batch["task_ids"].to(device)

    dec_in  = dec_ids[:, :-1]
    dec_tgt = dec_ids[:, 1:]
    dec_attn = (dec_in != pad_id).long()
    Tin = dec_in.size(1)

    with torch.no_grad():
        enc_out, enc_mask = model.encode(enc_ids, enc_attn=enc_attn, task_ids=task_ids)
        cache = model.new_cache()
        mixed = dec_in.clone()
        cur = dec_in[:, :1]  # position 0 (BOS) is always gold -- nothing to predict yet
        for t in range(1, Tin):
            step_out = model.decode_step(cur, enc_out, enc_ids, cache, position_offset=t - 1,
                                          enc_mask=enc_mask, use_copy=True)
            pred_t = step_out["log_probs"][:, -1].argmax(dim=-1)
            gold_t = dec_in[:, t]
            chosen = sampler.sample_or_teach(pred_t, gold_t, p_sample, pad_id)
            mixed[:, t] = chosen
            cur = chosen.unsqueeze(1)

    out = model(enc_ids, mixed, enc_attn=enc_attn, dec_attn=dec_attn,
                task_ids=task_ids, use_copy=True)
    return compute_loss(out, dec_tgt, pad_id, task_ids=task_ids)


def run_epoch(model, loader, optimizer, scheduler, device, pad_id: int,
              epoch=0, sampler=None, ss_enabled: bool = False, ss_mode: str = "parallel",
              grad_clip=1.0, log_every=100, nll_end_steps=50,
              checkpoint_every: int = 0, checkpoint_fn=None, resume_step: int = 0,
              loss_fn=compute_loss):
    """loss_fn: passed through to _step_batch (the non-scheduled-sampling path
    only -- the SS step functions are hardcoded to compute_loss). Defaults to
    compute_loss (S/M behavior, unchanged). Pass compute_loss_label_smoothed
    for the TLM-XS exposure-bias pilot.

    ss_mode selects which scheduled-sampling step function to use when
    ss_enabled and p_sample > 0: "parallel" (_step_batch_scheduled, the 2-pass
    approximation -- do NOT use with the copy pointer head) or "sequential"
    (_step_batch_scheduled_sequential, the copy-gate-safe autoregressive
    rollout). Default stays "parallel" for backward compatibility with
    existing call sites; it has no effect unless ss_enabled=True.

    checkpoint_every/checkpoint_fn: if checkpoint_every > 0, checkpoint_fn(step)
    is called every checkpoint_every steps (step is the 0-based in-epoch step
    index, called after that step's optimizer.step()) -- lets a long-running
    epoch (e.g. the sequential-SS pilot's ~4h/epoch at full scale) persist
    progress before a session drop, instead of only saving at epoch
    boundaries. What to save is entirely up to the caller-supplied
    checkpoint_fn (model/optimizer/scheduler state + whatever run-specific
    metadata that run's resume logic needs) -- run_epoch itself stays agnostic
    to checkpoint format, same as it already is for the epoch-boundary saves
    every existing caller does after run_epoch returns.

    resume_step: skip this many batches at the start of the epoch without
    forward/backward/optimizer.step() -- lets a mid-epoch resume pick back up
    near where a prior mid-epoch checkpoint left off (model/optimizer weights
    already reflect training through resume_step) instead of re-training that
    prefix a second time. The DataLoader still iterates (and its shuffle order
    won't exactly match the interrupted run's), so this is a cheap-skip, not a
    byte-exact resume -- but it avoids the wasted GPU forward/backward cost of
    redoing already-trained steps."""
    model.train()
    p_sample = sampler.get_sampling_prob(epoch) if (sampler is not None and ss_enabled) else 0.0
    tot_loss, tot_nll, tot_pgen, n = 0.0, 0.0, 0.0, 0
    end_nlls = []   # last nll_end_steps NLL values for end-of-epoch estimate
    for step, batch in enumerate(loader):
        if step < resume_step:
            continue
        if p_sample > 0.0:
            if ss_mode == "sequential":
                loss, nll, pgen = _step_batch_scheduled_sequential(model, batch, device, pad_id, sampler, p_sample)
            elif ss_mode == "parallel":
                loss, nll, pgen = _step_batch_scheduled(model, batch, device, pad_id, sampler, p_sample)
            else:
                raise ValueError(f"Unknown ss_mode: {ss_mode!r} (expected 'parallel' or 'sequential')")
        else:
            loss, nll, pgen = _step_batch(model, batch, device, pad_id, loss_fn=loss_fn)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tot_loss += loss.item(); tot_nll += nll.item(); tot_pgen += pgen.item(); n += 1
        end_nlls.append(nll.item())
        if len(end_nlls) > nll_end_steps:
            end_nlls.pop(0)
        if log_every and step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    step {step:5d} | loss {loss.item():.4f} | nll {nll.item():.4f} | p_gen {pgen.item():.3f} | "
                  f"p_samp {p_sample:.2f} | lr {lr:.2e}")
        if checkpoint_every and checkpoint_fn is not None and (step + 1) % checkpoint_every == 0:
            checkpoint_fn(step + 1)
    return {"loss": tot_loss / max(n, 1), "nll": tot_nll / max(n, 1),
            "nll_end": sum(end_nlls) / max(len(end_nlls), 1),
            "p_gen": tot_pgen / max(n, 1), "p_sample": p_sample}


def validate(model, loader, device, pad_id: int, log_every: int = 0, loss_fn=compute_loss):
    """loss_fn defaults to compute_loss (S/M behavior, unchanged); pass
    compute_loss_label_smoothed for the TLM-XS exposure-bias pilot, matching
    whatever loss_fn run_epoch was given for that run."""
    model.eval()
    tot_loss, tot_nll, tot_pgen, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss, nll, pgen = _step_batch(model, batch, device, pad_id, loss_fn=loss_fn)
            tot_loss += loss.item(); tot_nll += nll.item(); tot_pgen += pgen.item(); n += 1
            if log_every and n % log_every == 0:
                print(f"  val batch {n:3d}/{len(loader)} | nll {tot_nll/n:.4f} | p_gen {tot_pgen/n:.3f}")
    return {"loss": tot_loss / max(n, 1), "nll": tot_nll / max(n, 1), "p_gen": tot_pgen / max(n, 1)}


def make_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
