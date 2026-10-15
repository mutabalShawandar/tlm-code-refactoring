# -- Confidence-Calibration Remedy, Variant B --
#
# Variant A (calibration_training.py) shapes the GENERATION distribution itself
# with a Brier term, mixed with the gold teacher-forced loss as an anchor. Real
# run result (TLM-M, tlm_m_rl_calibrated_all_data): task mean_score regressed
# 5-12pp per task (spec's own ceiling was ~2pp) while the honest (p05_conf-based)
# free-decode calibration gap barely moved. Root cause identified after the run:
# its correctness label (token_correctness_labels, literal SequenceMatcher
# alignment to the ONE gold answer) actively mislabels legitimate non-gold-literal
# answers as "wrong" for rename/extract_method/deduplicate -- all three are
# explicitly name-agnostic in src/eval/scorers.py, so the Brier term was fighting
# real, scorer-certified-correct outputs on exactly the tasks with the most
# answer diversity (rename regressed worst: -0.125 mean_score, -12.1pp applied_ok).
#
# Variant B sidesteps the accuracy-regression failure mode entirely: a small
# confidence head sits on top of a FROZEN body and never touches the generation
# distribution, so task score cannot regress by construction. It also uses the
# REAL per-task scorer (src.eval.scorers.TASK_SCORERS + OK_SCORE_THRESHOLD) for
# its correctness label instead of literal token-match, so the "ok/wrong" signal
# agrees with what applied_ok/mean_score already mean everywhere else in this
# codebase.

import torch
import torch.nn as nn

from src.inference.generate import greedy_decode
from src.training.training import sr_role_masks
from src.data_set import apply_sr, join_lines, split_lines_by_newline

SRC_VERSION = "confidence-head-v1"
print(f"[confidence_head_training.py] SRC_VERSION = {SRC_VERSION}")


class ConfidenceHead(nn.Module):
    """Predicts P(this decode is correct) from the frozen decoder's last hidden
    state. One linear layer, sigmoid applied in the loss/inference call site
    (not here) so callers can choose logits vs probability as needed."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, dec_out):
        """dec_out: (B, T, d_model) -> (B, T) logits."""
        return self.linear(dec_out).squeeze(-1)


def freeze_body(model: nn.Module):
    """Sets requires_grad=False on every body parameter. Call once before
    training the confidence head. Forward passes through the body should NOT
    be wrapped in torch.no_grad() afterward -- autograd still needs to track
    the path from dec_out into conf_head's parameters; frozen leaves simply
    don't accumulate or receive gradient updates (cheaper than a no_grad
    forward + detach, and the optimizer only ever holds conf_head's params
    regardless)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def sample_ok_label(d, pred_dec_ids, before_ids, tokenizer, config, task_scorers, ok_thresholds):
    """Honest per-sample correctness verdict for a greedy rollout, using the
    REAL task scorer -- mirrors run_full_eval's is_ok computation in
    the SFT eval notebook (score_eval.ipynb) exactly (apply_sr -> decode -> task_scorers[ttype]
    -> score >= ok_thresholds[ttype]), so this training signal and the eval
    battery's applied_ok/mean_score agree on what "correct" means. Deliberately
    does NOT use literal token-match to gold (see module docstring above for why
    that mislabels legitimate answers on name-agnostic tasks).

    Returns (is_ok: float 0.0/1.0, score: float, pred_after: list[int] | None)."""
    ttype = d["type"]
    NL = config.special_token_ids["newline"]
    after_ids = tokenizer.encode(d["after"])
    gold_after = join_lines(split_lines_by_newline(after_ids, NL), NL)
    pred_after = apply_sr(before_ids, pred_dec_ids, config)

    before_text = tokenizer.decode(before_ids)
    pred_text = tokenizer.decode(pred_after) if pred_after is not None else None
    gold_text = tokenizer.decode(gold_after)

    score_result = task_scorers[ttype](before_text, pred_text, gold_text)
    score = score_result["score"]
    is_ok = 1.0 if score >= ok_thresholds.get(ttype, 0.999) else 0.0
    return is_ok, score, pred_after


def compute_confidence_head_loss(head_logits, is_ok_label, content_mask):
    """Brier loss between sigmoid(head_logits) and the sample's honest is_ok
    label, broadcast across every content-token position of that sample (the
    label is a per-SAMPLE verdict -- there is no clean per-token ground truth
    once a free-decode rollout has drifted from gold, same reasoning
    calibration_training.py's guardrails note applies to generation). Restricted
    to content tokens via sr_role_masks' content_mask (excludes
    <search>/<replace>/<eos>/PAD), same convention as compute_calibration_loss.

    head_logits: (B, T). is_ok_label: (B, T) float 0/1, already broadcast.
    content_mask: (B, T)."""
    conf = torch.sigmoid(head_logits)
    denom = content_mask.sum().clamp(min=1.0)
    per_tok = (conf - is_ok_label) ** 2
    return (per_tok * content_mask).sum() / denom


def confidence_head_step(model, conf_head, records, tokenizer, config, device, pad_id,
                         make_sft_sr, task_scorers, ok_thresholds, max_new_tokens=512):
    """One minibatch. Body must already be frozen via freeze_body(model) before
    the first call. Per record: greedy rollout (no_grad, frozen body) -> honest
    scorer-based is_ok label -> differentiable replay of the rollout's own
    tokens through the (still-frozen-but-gradient-tracked) body -> conf_head ->
    Brier loss against the broadcast label. Aggregates over the minibatch.

    Returns None if no record in the minibatch produced a usable rollout."""
    bos = config.special_token_ids["bos"]
    model.eval()
    conf_head.train()
    total, n = 0.0, 0
    diag = {"cal": 0.0}
    for d in records:
        res = make_sft_sr(d, tokenizer, config)
        if res is None:
            continue
        enc_ids, gold_dec_ids, task_id = res
        before_ids = tokenizer.encode(d["before"])

        # --- 1. rollout on the model's OWN greedy trajectory (no grad) ---
        with torch.no_grad():
            roll = greedy_decode(model, enc_ids, task_id, device,
                                 max_new_tokens=max_new_tokens, collect_conf=False)
        pred = roll["tokens"]
        if len(pred) < 3:
            continue
        pred_tgt = pred[1:]

        # --- 2. honest sample-level label from the real task scorer ---
        is_ok, score, _ = sample_ok_label(d, pred, before_ids, tokenizer, config,
                                          task_scorers, ok_thresholds)

        # --- 3. differentiable replay of the rollout -> conf_head -> Brier loss ---
        enc_t = torch.tensor(enc_ids, device=device).unsqueeze(0)
        dec_in_t = torch.tensor(pred[:-1], device=device).unsqueeze(0)
        dec_tgt_t = torch.tensor(pred_tgt, device=device).unsqueeze(0)
        task_t = torch.tensor([task_id], device=device)
        out = model(enc_t, dec_in_t,
                    enc_attn=torch.ones_like(enc_t),
                    dec_attn=(dec_in_t != pad_id).long(),
                    task_ids=task_t, use_copy=True)
        head_logits = conf_head(out["dec_out"])
        _, _, _, content_mask = sr_role_masks(dec_tgt_t, pad_id)
        label_t = torch.full_like(head_logits, float(is_ok))
        L_cal = compute_confidence_head_loss(head_logits, label_t, content_mask)

        total = total + L_cal
        diag["cal"] += float(L_cal.detach())
        n += 1

    if n == 0:
        return None
    return {"loss": total / n, "mean_cal": diag["cal"] / n, "n": n}
