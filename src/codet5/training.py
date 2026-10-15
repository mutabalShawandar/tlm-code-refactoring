# -- Train/eval step helpers for the HF-T5 SEARCH/REPLACE pipeline ------------
# See sr_text.py's module docstring: HF-T5-specific, kept separate from
# src/training (TLM's own compute_loss/run_epoch/validate) -- no imports
# either direction.
#
# Deliberately does NOT override decoder_start_token_id, pad_token_id, or
# eos_token_id on the loaded model/tokenizer config anywhere in this package
# -- unlike the own-pretrain (random-init) CodeT5 track, which has to define
# those conventions itself, this one loads real pretrained weights and must
# leave their already-learned special-token conventions untouched. `labels`
# passed to the model are exactly what the tokenizer produces for the target
# text (no manual BOS slicing) -- HF's own `_shift_right(labels)` (using
# `model.config.decoder_start_token_id`, whatever that pretrained checkpoint
# actually uses) builds the correct decoder input internally. This is the
# standard HF T5 fine-tuning recipe (`tokenizer(text_target=...).input_ids`
# fed straight into `labels=`), not something specific to this project.

from typing import List, Tuple

import torch

MAX_NEW_TOKENS = 512  # hard rule -- do not lower (300 caused ~24% malformed/truncated outputs)


def hf_step_batch(model, batch, pad_id, device, amp_dtype=None):
    enc_ids = batch["enc_ids"].to(device)
    enc_attn = batch["enc_attn"].to(device)
    dec_ids = batch["dec_ids"].to(device)

    labels = dec_ids.clone()
    labels[labels == pad_id] = -100

    if amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(input_ids=enc_ids, attention_mask=enc_attn, labels=labels)
    else:
        out = model(input_ids=enc_ids, attention_mask=enc_attn, labels=labels)
    return out.loss


@torch.no_grad()
def hf_validate(model, loader, pad_id, device, amp_dtype=None, log_every=200):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i, batch in enumerate(loader):
        loss = hf_step_batch(model, batch, pad_id, device, amp_dtype=amp_dtype)
        total_loss += float(loss.detach())
        n_batches += 1
        if (i + 1) % log_every == 0:
            print(f"    [val] {i+1}/{len(loader)}", flush=True)
    return total_loss / max(n_batches, 1)


def hf_run_epoch(model, loader, optimizer, scheduler, pad_id, device,
                  grad_accum_steps=1, amp_dtype=None, grad_clip=1.0, log_every=200):
    """Gradient-accumulation-aware, same shape as the pretrain/own-CodeT5
    notebooks' version -- see those for the accumulation-window rationale."""
    model.train()
    total_loss, n_micro = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        loss = hf_step_batch(model, batch, pad_id, device, amp_dtype=amp_dtype)
        (loss / grad_accum_steps).backward()
        total_loss += float(loss.detach())
        n_micro += 1

        if n_micro % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (i + 1) % log_every == 0:
            print(f"    step {i+1}/{len(loader)}  loss={total_loss/n_micro:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    if n_micro % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_micro, 1)


@torch.no_grad()
def hf_greedy_decode(model, tokenizer, enc_ids, device, max_new_tokens=MAX_NEW_TOKENS) -> List[int]:
    input_ids = torch.tensor([enc_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    out = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=max_new_tokens, num_beams=1, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return out[0].tolist()


def decode_target_text(tokenizer, ids: List[int]) -> Tuple[str, bool]:
    """Decode a (possibly BOS-led, EOS-terminated) generated id sequence back
    to text, dropping everything from the first EOS onward and a leading BOS
    if present -- keeps `<search>`/`<replace>`/task-marker tokens as literal
    text (skip_special_tokens=False) so parse_sr_pairs_text can find them.
    Returns (text, is_truncated) -- is_truncated True if no EOS was ever
    produced within max_new_tokens (same signal as the TLM ladder's
    `pred_dec[-1] != EOS` truncation check)."""
    eos_id = tokenizer.eos_token_id
    is_truncated = eos_id not in ids
    if not is_truncated:
        ids = ids[:ids.index(eos_id)]
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and ids and ids[0] == bos_id:
        ids = ids[1:]
    return tokenizer.decode(ids, skip_special_tokens=False), is_truncated
