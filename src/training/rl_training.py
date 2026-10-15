"""Self-critical sequence training (Rennie et al. 2017) for TLM-S SFT policies.

Motivation: scheduled sampling was tried once already (TLM-XS,
"Try #3") and failed -- splicing the model's own predicted token ids into the
decoder *input* mid-sequence corrupts the copy-pointer's cross-attention,
since it's now conditioning on a hybrid sequence it would never have produced
coherently itself. RL avoids this structurally: the model generates a
complete sequence through its own real decode path (sample_decode /
greedy_decode, unmodified), and only the *log-probability* of that already-
complete, self-consistent sequence is adjusted afterwards via the reward.
Nothing is spliced into the input the copy mechanism reads.

Self-critical sequence training specifically (over vanilla REINFORCE): uses
the model's own greedy decode of the same input as the reward baseline
(instead of a learned value network), so the advantage
`reward(sample) - reward(greedy)` is positive only when stochastic
exploration actually beat the policy's own current best guess. No extra
network, no separate baseline training loop.

MLE-mixed loss (added after pilot runs 1-3 all collapsed by epoch 2-3, most
recently even with the p_gen anchor from below): pure REINFORCE has nothing
anchoring the policy back toward its SFT optimum, and pilots showed negative
mean_advantage from step 1 onward (sampled rollouts consistently scored worse
than the policy's own greedy decode) -- noisy exploration with no pull-back
compounds into collapse. Standard fix from the literature (Ranzato et al.
2016 "MIXER"; Paulus et al. 2017): mix the RL loss with the original
teacher-forced supervised loss on the same record's gold target,
`loss = gamma * reinforce_loss + (1 - gamma) * mle_nll`. This does not
reopen Try #3's failure mode -- the MLE term is an ordinary teacher-forced
forward pass on the *gold* target (exactly what SFT already does), computed
alongside the sampled rollout's REINFORCE term, not spliced into anything the
sampled rollout's decode saw. `gamma=1.0` (default) reproduces the old
pure-REINFORCE behavior exactly, for backward compatibility.
"""

import torch
import torch.nn.functional as F

from src.inference.generate import greedy_decode, sample_decode
from src.training.training import (
    PGEN_SEARCH, PGEN_REPLACE, PGEN_NEUTRAL, PGEN_LAMBDA, sr_role_masks,
)

# training.py's PGEN_LAMBDA=1.0 was tuned against the NLL-scale SFT/pretrain
# loss (typically O(1-3) nats). RL's own loss term is
# -advantage * mean_logprob, which shrinks toward zero as the policy
# converges (small reward_sample - reward_greedy times a per-token log-prob)
# -- reusing PGEN_LAMBDA=1.0 unscaled risks the anchor dominating/drowning
# the actual reward gradient instead of just preventing copy-gate collapse.
# Separate, RL-specific constant so it can be tuned without touching the
# SFT/pretrain value; self_critical_step's pgen_lambda= param can override
# this per-run from the notebook.
RL_PGEN_LAMBDA = 0.1

# Confidence-reward consistency term (see project_rl_mle_mixed_epoch1_result
# memory, "planned next step"): none of the existing loss terms target
# exposure-bias calibration -- REINFORCE optimizes task score via reward, the
# MLE term reinforces the already-well-calibrated teacher-forced regime. This
# term operates on the *sampled* (free-decode-like) rollout specifically,
# where the miscalibration actually lives, using log_probs the pipeline
# already computes -- no new decode pass. Same order-of-magnitude reasoning
# as RL_PGEN_LAMBDA: this loss's own scale (a bounded [0,1] squared error) is
# comparable to the REINFORCE term's typical magnitude, not to training.py's
# NLL-scale loss, so it gets its own small constant rather than reusing either.
CONF_LAMBDA = 0.1


def _teacher_forced_logprob_sum(model, enc_ids, tokens, task_id, device, pad_id):
    """Single parallel forward pass over an already-decoded token sequence,
    returning (sum of log p(token_t | token_<t) for content tokens, n_content).

    This is what makes RL here cheap: the sequence was already generated
    (sample_decode / greedy_decode, no_grad), so getting its gradient-bearing
    log-probability doesn't require backpropagating through the autoregressive
    loop -- one ordinary teacher-forced forward pass (same shape as
    training.py::_step_batch) replays it and reads off the log-prob of each
    token the model actually chose. Mathematically exact, not an approximation:
    decode is deterministic given the encoder + prefix, so teacher-forcing the
    model's own emitted sequence reproduces the same per-step distributions it
    used to generate it (eval mode, no dropout)."""
    enc_t = torch.tensor(enc_ids, dtype=torch.long, device=device).unsqueeze(0)
    enc_attn = torch.ones_like(enc_t)
    task_t = torch.tensor([task_id], dtype=torch.long, device=device)
    dec_t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    dec_in = dec_t[:, :-1]
    dec_tgt = dec_t[:, 1:]
    dec_attn = torch.ones_like(dec_in)

    out = model(enc_t, dec_in, enc_attn=enc_attn, dec_attn=dec_attn,
                task_ids=task_t, use_copy=True)
    log_probs = out["log_probs"]                                   # (1, T, V)
    tok_logprob = log_probs.gather(-1, dec_tgt.unsqueeze(-1)).squeeze(-1)  # (1, T)
    mask = (dec_tgt != pad_id).float()
    logprob_sum = (tok_logprob * mask).sum()
    n_content = mask.sum().clamp(min=1.0)
    return logprob_sum, n_content, out


def _mean_sampled_token_confidence(log_probs, dec_tgt, mask):
    """Differentiable mean probability the model assigned to the token it
    actually SAMPLED at each position, not the argmax/top-1 token -- these
    differ whenever sample_decode's temperature sampling (temperature=0.5,
    not greedy) picked a non-argmax token, which is common enough that
    conf_pen (mean_conf - reward_sample)**2 was previously penalizing
    confidence in a token the model didn't necessarily choose, weakening the
    signal independent of CONF_LAMBDA. `mean_conf`/mean_top1_confidence's old
    name suggested "top-1" deliberately matched generate.py::greedy_decode's
    `mean_conf` (argmax probability, correct there since greedy_decode always
    picks the argmax) -- but self_critical_step's rollout comes from
    sample_decode, where argmax and the sampled token routinely diverge, so
    that definition was never actually the right one to reuse here. `dec_tgt`
    is the decoder target sequence (i.e. the sampled tokens themselves,
    shifted) -- same tensor _teacher_forced_logprob_sum already gathers
    log-probs against for the REINFORCE term, just not previously threaded
    through to this function."""
    sampled_logprob = log_probs.gather(-1, dec_tgt.unsqueeze(-1)).squeeze(-1)  # (1, T)
    sampled_prob = sampled_logprob.exp()
    return (sampled_prob * mask).sum() / mask.sum().clamp(min=1.0)


def sample_and_score(model, d, tokenizer, config, device, task_scorers,
                      make_sft_sr, apply_sr, split_lines_by_newline, join_lines,
                      temperature=1.0, max_new_tokens=512):
    """One rollout: greedy baseline + stochastic sample, both scored with the
    real continuous task scorer (src/eval/scorers.py). Returns None if the
    record can't be encoded (mirrors SFTDataset's own drop behavior) or the
    sampled sequence is degenerate (empty content -- nothing to train on).

    `make_sft_sr`/`apply_sr`/`split_lines_by_newline`/`join_lines` are passed
    in rather than imported (same "pass paths/deps in, don't hardcode" spirit
    as the rest of src/ -- keeps this module decoupled from data_set's
    tokenizer-corpus-specific helpers at import time)."""
    res = make_sft_sr(d, tokenizer, config)
    if res is None:
        return None
    enc_ids, gold_dec_ids, task_id = res
    before_ids = tokenizer.encode(d["before"])
    after_ids = tokenizer.encode(d["after"])
    nl_id = config.special_token_ids["newline"]
    gold_after = join_lines(split_lines_by_newline(after_ids, nl_id), nl_id)

    before_text = tokenizer.decode(before_ids)
    gold_text = tokenizer.decode(gold_after)
    scorer = task_scorers[d["type"]]

    greedy_out = greedy_decode(model, enc_ids, task_id, device,
                                max_new_tokens=max_new_tokens, collect_conf=False)
    greedy_pred = apply_sr(before_ids, greedy_out["tokens"], config)
    greedy_text = tokenizer.decode(greedy_pred) if greedy_pred is not None else None
    reward_greedy = scorer(before_text, greedy_text, gold_text)["score"]

    sample_out = sample_decode(model, enc_ids, task_id, device,
                                max_new_tokens=max_new_tokens, temperature=temperature)
    sample_tokens = sample_out["tokens"]
    if len(sample_tokens) < 2:
        return None
    sample_pred = apply_sr(before_ids, sample_tokens, config)
    sample_text = tokenizer.decode(sample_pred) if sample_pred is not None else None
    reward_sample = scorer(before_text, sample_text, gold_text)["score"]

    return {
        "enc_ids": enc_ids, "task_id": task_id, "sample_tokens": sample_tokens,
        "reward_sample": reward_sample, "reward_greedy": reward_greedy,
        "gold_dec_ids": gold_dec_ids,
    }


# -- CPU-only parallel rollout collection ------------------------------------
# sample_and_score's decode (greedy_decode/sample_decode, both @torch.no_grad
# internally) is read-only and fully independent per record -- an
# embarrassingly-parallel dispatch, not a change to the gradient/optimizer
# path (self_critical_step below still runs sequentially, unchanged, on
# whatever a collector returns). This is the CPU-native equivalent of GPU
# batching: CPU has no "batching is nearly free" headroom to unlock the way a
# GPU does, so process-level parallelism across cores is the lever here
# instead.
#
# NOT using torch.multiprocessing's model.share_memory() "Hogwild" pattern,
# despite that being the textbook approach for this: empirically broken on
# this project's actual environment (Windows + Python 3.13 + torch 2.12.1
# CPU) -- every worker crashed at bootstrap with
# `RuntimeError: Couldn't open shared event ..., error code: <2>` trying to
# reconstruct the shared-memory tensor handle. Confirmed via a standalone
# test script before wiring this into any real training loop, not assumed.
# Instead, each worker holds its OWN local TLM_S instance (constructed once
# in _pool_worker_init) and gets the current weights pushed to it via plain
# state_dict() pickling -- ordinary CPU tensors, no shared-memory handles
# involved, portable across platforms. Cost: one state_dict broadcast per
# .collect() call (tens of ms for a ~6M-param model), not per record --
# negligible next to the decode cost this is meant to parallelize.
#
# CPU only, deliberately: CUDA + multiprocessing needs a "spawn" start method
# plus a CUDA context per worker process, and none of the above applies the
# same way -- for a model this small the extra complexity isn't worth it.
# ParallelRolloutCollector asserts CPU rather than silently doing the wrong
# (or nothing) with a GPU model.
#
# Windows-notebook caveat (unrelated to the share_memory() issue above, still
# applies): multiprocessing on Windows only supports the "spawn" start
# method, which requires worker entry points to be real importable
# module-level functions -- _pool_worker_init/_pool_worker_set_weights/
# _pool_worker_call below are (that's why they're plain module functions, not
# closures/lambdas) -- but the *call site* that constructs a
# ParallelRolloutCollector must sit behind `if __name__ == "__main__":` or
# equivalent, or spawned child processes can misbehave trying to re-bootstrap
# a notebook kernel. Colab's Linux backend supports "fork" and doesn't have
# this restriction.

_worker_state = {}  # populated once per worker process by _pool_worker_init, never in the parent


def _pool_worker_init(config, tokenizer_path, task_scorers,
                       make_sft_sr, apply_sr, split_lines_by_newline, join_lines,
                       temperature, max_new_tokens):
    """Pool(initializer=...) target -- runs once per worker process, not once
    per task, so the model/tokenizer/scorers/helpers are set up a single time
    and reused across every record that worker ever handles.

    Builds its own local TLM_S(config) rather than receiving the parent's
    model object -- see the module-level comment above for why (Windows
    share_memory() breakage). Starts with random init weights; real weights
    arrive via _pool_worker_set_weights before the first real .collect() call
    (ParallelRolloutCollector.collect always broadcasts first).

    `tokenizer_path` (not a live tokenizer object) is passed and reloaded
    fresh here, sidestepping any question of whether the HuggingFace Rust
    tokenizer pickles cleanly across the process boundary -- same convention
    every other module in this codebase already uses for tokenizer setup, not
    something new invented for this."""
    from src.tokenizer import TLM_Tokenizer
    from src.model_variants import TLM_S

    model = TLM_S(config)
    model.eval()
    tok = TLM_Tokenizer(vocab_size=config.vocab_size)
    tok.load(tokenizer_path)

    _worker_state.update({
        "model": model, "tokenizer": tok, "config": config,
        "task_scorers": task_scorers,
        "make_sft_sr": make_sft_sr, "apply_sr": apply_sr,
        "split_lines_by_newline": split_lines_by_newline, "join_lines": join_lines,
        "temperature": temperature, "max_new_tokens": max_new_tokens,
    })


def _pool_worker_set_weights(state_dict):
    """Refreshes this worker's local model with the parent's current weights.
    Plain state_dict() pickling (ordinary CPU tensors) -- see module-level
    comment for why this replaces share_memory()."""
    _worker_state["model"].load_state_dict(state_dict)
    return True


def _pool_worker_call(d):
    """Pool.imap_unordered per-task entry point -- runs sample_and_score
    against this worker's process-local state (set up once by
    _pool_worker_init, weights refreshed by _pool_worker_set_weights before
    each batch), always on CPU (see module-level comment for why this isn't
    wired up for CUDA)."""
    s = _worker_state
    return sample_and_score(s["model"], d, s["tokenizer"], s["config"],
                             torch.device("cpu"), s["task_scorers"],
                             s["make_sft_sr"], s["apply_sr"],
                             s["split_lines_by_newline"], s["join_lines"],
                             temperature=s["temperature"],
                             max_new_tokens=s["max_new_tokens"])


class ParallelRolloutCollector:
    """Persistent worker-process pool for CPU-only parallel rollout
    collection. See the module-level comment above this class for the full
    design rationale (why CPU multiprocessing instead of GPU-style batching,
    why state_dict-broadcast instead of share_memory(), and the
    Windows/notebook __main__-guard caveat).

    Usage: construct once per run (not per epoch/step -- construction pays
    process-spawn cost), call .collect(records) per batch (broadcasts current
    weights first, then dispatches), call .close() when done (or use as a
    context manager). Pass the *live* model in -- .collect() reads its
    current state_dict() fresh on every call, so ordinary optimizer.step()
    calls between .collect() invocations are picked up automatically without
    the caller having to remember to push weights explicitly."""

    def __init__(self, model, tokenizer_path, config, task_scorers,
                 make_sft_sr, apply_sr, split_lines_by_newline, join_lines,
                 temperature, max_new_tokens, n_workers):
        assert next(model.parameters()).device.type == "cpu", (
            "ParallelRolloutCollector is CPU-only -- see class/module docstring")
        self.model = model
        self.n_workers = n_workers
        import torch.multiprocessing as _mp
        self.pool = _mp.Pool(
            processes=n_workers,
            initializer=_pool_worker_init,
            initargs=(config, tokenizer_path, task_scorers,
                      make_sft_sr, apply_sr, split_lines_by_newline, join_lines,
                      temperature, max_new_tokens),
        )

    def _broadcast_weights(self):
        # map with exactly n_workers identical tasks reliably lands one task
        # per idle worker process in CPython's Pool implementation when task
        # count == worker count -- not a documented guarantee, but the
        # standard practical idiom for "run this once on every worker" since
        # multiprocessing.Pool has no true broadcast primitive. Worst case if
        # it were ever violated: a worker decodes one batch on stale weights,
        # not a correctness break for RL (already noise-tolerant), so this is
        # an acceptable trade for avoiding share_memory() entirely.
        state_dict = self.model.state_dict()
        self.pool.map(_pool_worker_set_weights, [state_dict] * self.n_workers)

    def collect(self, records):
        """records -> list of sample_and_score results, None entries dropped
        (same drop convention sample_and_score itself uses when a record
        can't be encoded or its sampled sequence is degenerate). Order is not
        guaranteed to match `records`' order (imap_unordered) -- fine here
        since self_critical_step aggregates a batch order-independently."""
        self._broadcast_weights()
        results = []
        for r in self.pool.imap_unordered(_pool_worker_call, records):
            if r is not None:
                results.append(r)
        return results

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def self_critical_step(model, rollouts, device, pad_id, ref_model=None, kl_coef=0.0,
                        pgen_lambda=RL_PGEN_LAMBDA, gamma=1.0, conf_lambda=CONF_LAMBDA):
    """Aggregates a list of sample_and_score() results into one policy-gradient
    loss (mean over the batch), replaying each sampled sequence through
    _teacher_forced_logprob_sum for its gradient-bearing log-prob.

    reinforce_i = -(reward_sample_i - reward_greedy_i) * mean_logprob_i
    (advantage-weighted log-likelihood; mean, not sum, over content tokens so
    longer samples don't get an automatically larger gradient magnitude just
    from length -- same length-normalization convention as NLL elsewhere in
    this codebase).

    `pgen_lambda` defaults to RL_PGEN_LAMBDA (0.1), not training.py's
    PGEN_LAMBDA (1.0) -- see the module-level comment on RL_PGEN_LAMBDA.
    Pass pgen_lambda=0 to disable the anchor entirely.

    `gamma` (default 1.0, i.e. pure REINFORCE, old behavior unchanged) mixes
    in a teacher-forced MLE term on the record's own gold SFT target:
      loss_i = gamma * reinforce_i + (1 - gamma) * mle_nll_i
    See the module docstring for why this was added. mle_nll_i is computed by
    replaying the gold `dec_ids` (not the sampled tokens) through the same
    _teacher_forced_logprob_sum helper used for the REINFORCE term, so it's an
    ordinary SFT-style teacher-forced NLL, not something new invented for RL.
    Pass gamma=1.0 (or omit) to skip this computation entirely and reproduce
    pilot runs 1-3's exact loss.

    Optional KL term against a frozen `ref_model` (e.g. the pre-RL SFT+coverage
    checkpoint) discourages the policy from drifting into reward-hacking
    degenerate output that happens to score well under the imperfect scorer
    but reads as nonsense -- set kl_coef=0.0 to disable (default; enable only
    after confirming the base self-critical loss doesn't already collapse the
    copy gate the way Try #3's scheduled sampling did -- see module docstring).

    `conf_lambda` (default CONF_LAMBDA=0.1) weights a confidence-reward
    consistency term computed on the *sampled* rollout:
      conf_pen_i = (mean_sampled_token_confidence(sample_i) - reward_sample_i) ** 2
    Directly targets exposure bias (see project_rl_mle_mixed_epoch1_result
    memory): the free-decode calibration gap persisted through the MLE-mixed
    run because nothing in that loss related the model's own confidence on
    its self-generated trajectory to how good that trajectory actually was.
    This does -- pushes confidence down on low-reward samples and permits it
    up on high-reward ones, using the same log_probs tensor already computed
    for the REINFORCE term (no extra forward/decode pass). Pass
    conf_lambda=0 to disable."""
    if not rollouts:
        return None

    # Defensive, not just a caller convention: sample_and_score's rollouts (just
    # collected via greedy_decode/sample_decode, both of which force model.eval()
    # internally) leave `model` in eval mode. Without this, the gradient-bearing
    # forward pass below would silently run with dropout disabled every step,
    # even after the training loop's own model.train() call -- same
    # defensive-mode-setting convention as training.py's run_epoch/validate.
    model.train()

    total_loss = 0.0
    total_reinforce_loss = 0.0
    total_mle_nll = 0.0
    total_kl = 0.0
    total_pgen_penalty = 0.0
    total_conf_penalty = 0.0
    total_mean_conf = 0.0
    advantages, rewards_sample, rewards_greedy = [], [], []

    for r in rollouts:
        logprob_sum, n_content, out = _teacher_forced_logprob_sum(
            model, r["enc_ids"], r["sample_tokens"], r["task_id"], device, pad_id)
        mean_logprob = logprob_sum / n_content
        advantage = r["reward_sample"] - r["reward_greedy"]
        reinforce_term = -advantage * mean_logprob
        total_loss = total_loss + gamma * reinforce_term
        total_reinforce_loss += float(reinforce_term.detach())

        if gamma < 1.0:
            gold_logprob_sum, gold_n_content, _ = _teacher_forced_logprob_sum(
                model, r["enc_ids"], r["gold_dec_ids"], r["task_id"], device, pad_id)
            mle_nll = -(gold_logprob_sum / gold_n_content)
            total_loss = total_loss + (1.0 - gamma) * mle_nll
            total_mle_nll += float(mle_nll.detach())

        # Ported from training.py::compute_loss: RL had nothing anchoring p_gen,
        # so it could drift to a degenerate copy/generate balance independent of
        # LR/KL (see reward_greedy collapse 0.626->0.309 within epoch 1).
        p_gen = out.get("p_gen")
        if p_gen is not None and pgen_lambda > 0:
            dec_t = torch.tensor(r["sample_tokens"], dtype=torch.long, device=device).unsqueeze(0)
            dec_tgt = dec_t[:, 1:]
            in_search, in_replace, _, content_mask = sr_role_masks(dec_tgt, pad_id)
            pgen_target = torch.full_like(p_gen, PGEN_NEUTRAL)
            pgen_target[in_search]  = PGEN_SEARCH
            pgen_target[in_replace] = PGEN_REPLACE
            denom_c = content_mask.sum().clamp(min=1.0)
            pgen_pen = (((p_gen - pgen_target) ** 2) * content_mask).sum() / denom_c
            total_loss = total_loss + pgen_lambda * pgen_pen
            total_pgen_penalty += float(pgen_pen.detach())

        if conf_lambda > 0:
            dec_t = torch.tensor(r["sample_tokens"], dtype=torch.long, device=device).unsqueeze(0)
            dec_tgt_conf = dec_t[:, 1:]
            # Role-aware content mask (same sr_role_masks used for the p_gen
            # anchor above), not a plain pad-mask -- excludes <search>/
            # <replace>/<eos>. Added after pilot conf_lambda=0.1 showed
            # conf_sample flat across an epoch+ despite reward_sample staying
            # well below it: pretraining's span-corruption objective teaches
            # (correctly) very high confidence on structural/copy-heavy
            # tokens, and averaging those into the confidence term dilutes
            # whatever signal exists at the few actually-uncertain content
            # positions this term is meant to target. This does not by
            # itself fix a too-small CONF_LAMBDA; it's a separate, orthogonal
            # fix to what the penalty is actually averaged over.
            _, _, _, content_mask_conf = sr_role_masks(dec_tgt_conf, pad_id)
            # Confidence in the token actually SAMPLED, not the argmax --
            # see _mean_sampled_token_confidence's docstring. This is the
            # fix for the argmax-vs-sampled-token mismatch identified after
            # conf_sample stayed flat across 3 epochs (two CONF_LAMBDA
            # values) despite reward_sample staying well below it -- raising
            # the weight further wasn't going to help if the term was
            # measuring confidence in a token distribution property (argmax)
            # rather than confidence in the model's actual choice.
            mean_conf = _mean_sampled_token_confidence(out["log_probs"], dec_tgt_conf, content_mask_conf)
            conf_pen = (mean_conf - r["reward_sample"]) ** 2
            total_loss = total_loss + conf_lambda * conf_pen
            total_conf_penalty += float(conf_pen.detach())
            total_mean_conf += float(mean_conf.detach())

        if ref_model is not None and kl_coef > 0:
            with torch.no_grad():
                _, _, ref_out = _teacher_forced_logprob_sum(
                    ref_model, r["enc_ids"], r["sample_tokens"], r["task_id"], device, pad_id)
            # per-token KL(policy || ref) on the sampled sequence, content tokens only
            enc_t = torch.tensor(r["enc_ids"], dtype=torch.long, device=device).unsqueeze(0)
            dec_t = torch.tensor(r["sample_tokens"], dtype=torch.long, device=device).unsqueeze(0)
            dec_tgt = dec_t[:, 1:]
            mask = (dec_tgt != pad_id).float()
            policy_lp = out["log_probs"]
            ref_lp = ref_out["log_probs"]
            kl_per_tok = (policy_lp.exp() * (policy_lp - ref_lp)).sum(-1)
            kl = (kl_per_tok * mask).sum() / mask.sum().clamp(min=1.0)
            total_loss = total_loss + kl_coef * kl
            total_kl += float(kl.detach())

        advantages.append(advantage)
        rewards_sample.append(r["reward_sample"])
        rewards_greedy.append(r["reward_greedy"])

    n = len(rollouts)
    return {
        "loss": total_loss / n,
        "mean_advantage": sum(advantages) / n,
        "mean_reward_sample": sum(rewards_sample) / n,
        "mean_reward_greedy": sum(rewards_greedy) / n,
        "mean_kl": total_kl / n,
        "mean_pgen_penalty": total_pgen_penalty / n,
        # confidence-reward consistency diagnostics: mean_confidence_sample vs
        # mean_reward_sample (both above) is the same axis as the eval
        # notebook's free-decode calibration gap -- watch these converge
        # during training as a live signal the term is doing its job.
        "mean_confidence_sample": total_mean_conf / n,
        "mean_conf_penalty": total_conf_penalty / n,
        "mean_conf_term_weighted": conf_lambda * total_conf_penalty / n,
        # unweighted REINFORCE term and the weighted pgen-penalty term, reported
        # separately (not just fused into "loss") so a run can be diagnosed for
        # the pgen anchor dominating/drowning the actual reward gradient --
        # PGEN_LAMBDA was tuned against training.py's NLL-scale loss, not RL's
        # much-smaller advantage-scale loss, and that mismatch was never verified.
        "mean_reinforce_loss": total_reinforce_loss / n,
        "mean_pgen_term_weighted": pgen_lambda * total_pgen_penalty / n,
        # unweighted gold-target teacher-forced NLL and its gamma-weighted
        # contribution to the loss -- 0.0 when gamma=1.0 (MLE mixing disabled).
        "mean_mle_nll": total_mle_nll / n,
        "mean_mle_term_weighted": (1.0 - gamma) * total_mle_nll / n,
    }
