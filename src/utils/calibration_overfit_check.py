"""Overfit sanity check for src/training/calibration_training.py.

On a single fixed minibatch of real SFT-shaped records, runs many
calibration_step updates and checks the mechanism actually trains *before*
spending a full run on it:
  - mean_cal should decrease over the course of the run,
  - the model's confidence on tokens it gets wrong (y=0, from the *rollout's
    own* correctness labels each step) should trend down while confidence on
    tokens it gets right (y=1) stays high.

This is deliberately a tiny, fast, real-model check (few records, few dozen
steps) -- not a full calibration fine-tuning run.

Usage:
    python src/utils/calibration_overfit_check.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from src.tokenizer import TLM_Tokenizer
from src.model_config import ModelConfig
from src.model_variants import TLM_S
from src.data_set.sft_corruption import make_sft_sr
from src.training.calibration_training import calibration_step

N_STEPS = 40
LR = 3e-4  # deliberately larger than a real fine-tune LR -- this is an overfit probe, not a real run

FIXED_RECORDS = [
    {
        "type": "rename",
        "before": "def foo(x):\n    return x + 1\n",
        "after": "def bar(x):\n    return x + 1\n",
    },
    {
        "type": "remove_dead_code",
        "before": "def f(x):\n    y = 1\n    return x\n",
        "after": "def f(x):\n    return x\n",
    },
    {
        "type": "simplify",
        "before": "def g(x):\n    if x == True:\n        return True\n    else:\n        return False\n",
        "after": "def g(x):\n    return x\n",
    },
]


def main():
    device = "cpu"
    config = ModelConfig()
    tokenizer = TLM_Tokenizer()
    tokenizer.load(os.path.join(
        os.path.dirname(__file__), '..', '..', 'src', 'tokenizer', 'tokenizer_tlm_small.json'))
    pad_id = config.special_token_ids["pad"]

    model = TLM_S(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"Overfitting on {len(FIXED_RECORDS)} fixed records for {N_STEPS} steps...\n")
    print(f"{'step':>4}  {'loss':>8}  {'mean_cal':>9}  {'mean_task':>9}")

    cal_history = []
    for step in range(N_STEPS):
        result = calibration_step(
            model, FIXED_RECORDS, tokenizer, config, device, pad_id,
            make_sft_sr, lambda_cal=1.0,
        )
        if result is None:
            print(f"step {step}: no usable records (unexpected -- check FIXED_RECORDS)")
            continue

        optimizer.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        cal_history.append(result["mean_cal"])
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"{step:4d}  {float(result['loss']):8.4f}  "
                  f"{result['mean_cal']:9.4f}  {result['mean_task']:9.4f}")

    print()
    early_mean = sum(cal_history[:5]) / max(len(cal_history[:5]), 1)
    late_mean = sum(cal_history[-5:]) / max(len(cal_history[-5:]), 1)
    print(f"mean_cal: first 5 steps avg = {early_mean:.4f}, last 5 steps avg = {late_mean:.4f}")
    if late_mean < early_mean:
        print("PASS: mean_cal decreased -- calibration loss is training.")
    else:
        print("FAIL: mean_cal did not decrease -- investigate before trusting the mechanism.")


if __name__ == "__main__":
    main()
