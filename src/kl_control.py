"""
kl_control.py
-------------
Counter-training ablation: tests whether KL divergence regularization
(beta in GRPOConfig) is responsible for the frugal model's resistance to
behavioral flipping.

Runs the same counter-training procedure as counter_train.py but with
beta=0.0 (no KL penalty), then compares flip steps to the original CT results.

Key question:
  - Original CT (beta=0.05): frugal never flips, baseline flips at ~275
  - No-KL CT  (beta=0.0):   does frugal still resist?  Does baseline flip earlier?

If frugal still resists → robustness comes from learned representation, not KL.
If frugal flips faster → KL was the stabilizer (representational argument weakens).

Usage:
    python kl_control.py \
        --frugal_path   outputs/frugal_full/final \
        --baseline_path outputs/baseline_full/final \
        --original_ct   outputs/counter_training
"""

import argparse
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import score_first_match
from counter_train import (
    make_verbosity_reward_fn,
    measure_behavior,
    BehavioralCheckpoint,
)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_no_kl_ct(
    model_path:       str,
    model_label:      str,
    train_tasks:      list[dict],
    probe_tasks:      list[dict],
    output_dir:       Path,
    n_steps:          int   = 300,
    behavioral_every: int   = 25,
    batch_size:       int   = 8,
    lr:               float = 1e-5,
    alpha:            float = 0.3,
    max_tokens:       int   = 150,
    temperature:      float = 0.7,
    num_generations:  int   = 4,
    seed:             int   = 42,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading {model_label} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Measure baseline tokens for flip threshold
    ckpt = measure_behavior(
        model, tokenizer, probe_tasks, alpha, max_tokens,
        flip_threshold=9999.0,  # dummy — just want mean_tokens
        device=device,
    )
    ckpt.step = 0
    baseline_tokens = ckpt.mean_tokens
    flip_threshold  = baseline_tokens * 2.0
    print(f"  [{model_label}] Pre-CT: mean_tokens={baseline_tokens:.1f}, "
          f"flip_threshold={flip_threshold:.1f}")

    behavioral_log = [ckpt]
    flip_step: Optional[int] = None

    # Dataset
    records = [
        {"prompt": [{"role": "user", "content": t["prompt"]}],
         "ground_truths": t["answer"]}
        for t in train_tasks
    ]
    dataset = Dataset.from_list(records)
    reward_fn = make_verbosity_reward_fn(alpha=alpha, max_tokens=max_tokens)

    ckpt_dir = output_dir / model_label / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir           = str(ckpt_dir / "trainer_tmp"),
        num_train_epochs     = 1,
        max_steps            = behavioral_every,
        per_device_train_batch_size = batch_size,
        learning_rate        = lr,
        max_completion_length = max_tokens,
        num_generations      = num_generations,
        temperature          = temperature,
        seed                 = seed,
        save_steps           = behavioral_every + 1,
        logging_steps        = behavioral_every,
        report_to            = "none",
        beta                 = 0.0,   # ← key ablation: no KL penalty
        loss_type            = "grpo",
    )

    steps_done = 0
    segment_size = behavioral_every

    while steps_done < n_steps:
        steps_this_segment = min(segment_size, n_steps - steps_done)
        grpo_config.max_steps = steps_this_segment

        trainer = GRPOTrainer(
            model            = model,
            reward_funcs     = reward_fn,
            args             = grpo_config,
            train_dataset    = dataset,
            processing_class = tokenizer,
        )
        trainer.train()

        steps_done += steps_this_segment

        save_path = ckpt_dir / f"checkpoint-{steps_done}"
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        ckpt = measure_behavior(
            model, tokenizer, probe_tasks, alpha, max_tokens,
            flip_threshold, device,
        )
        ckpt.step = steps_done
        behavioral_log.append(ckpt)

        print(f"  [{model_label}] Step {steps_done}: "
              f"mean_tokens={ckpt.mean_tokens:.1f}, "
              f"accuracy={ckpt.accuracy:.2f}, "
              f"flipped={ckpt.flipped}")

        if ckpt.flipped and flip_step is None:
            flip_step = steps_done
            print(f"  *** FLIP at step {steps_done} ***")

    return {
        "model":             model_label,
        "flip_step":         flip_step,
        "baseline_tokens":   baseline_tokens,
        "behavioral_log":    [asdict(c) for c in behavioral_log],
    }


# ── Load original CT results for comparison ───────────────────────────────────

def load_original_flip_step(original_ct_dir: str, model_label: str) -> Optional[int]:
    log_path = Path(original_ct_dir) / model_label / "behavioral_log.json"
    if not log_path.exists():
        return None
    with open(log_path) as f:
        log = json.load(f)
    for entry in log:
        if entry.get("flipped"):
            return entry["step"]
    return None  # never flipped


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frugal_path",   required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--original_ct",   required=True,
                        help="Path to original counter_training output dir")
    parser.add_argument("--data_path",      default="data/train.json")
    parser.add_argument("--probe_data_path", default="data/val.json")
    parser.add_argument("--output_dir",    default="outputs/kl_control")
    parser.add_argument("--n_steps",       type=int, default=300)
    parser.add_argument("--alpha",         type=float, default=0.3)
    parser.add_argument("--skip_frugal",   action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    with open(args.data_path)      as f: train_tasks = json.load(f)
    with open(args.probe_data_path) as f: probe_tasks = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, model_path, skip in [
        ("frugal",   args.frugal_path,   args.skip_frugal),
        ("baseline", args.baseline_path, args.skip_baseline),
    ]:
        if skip:
            print(f"  [skip] {model_name}")
            continue

        result = run_no_kl_ct(
            model_path       = model_path,
            model_label      = model_name,
            train_tasks      = train_tasks,
            probe_tasks      = probe_tasks,
            output_dir       = output_dir,
            n_steps          = args.n_steps,
            alpha            = args.alpha,
        )
        all_results[model_name] = result

        log_path = output_dir / model_name / "behavioral_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(result["behavioral_log"], f, indent=2)

    # Comparison table
    print("\n" + "="*60)
    print("  KL CONTROL COMPARISON")
    print("="*60)
    print(f"  {'model':>10}  {'original flip':>15}  {'no-KL flip':>12}")
    print("  " + "-"*40)

    for model_name in ["frugal", "baseline"]:
        orig_flip = load_original_flip_step(args.original_ct, model_name)
        no_kl_flip = (all_results.get(model_name) or {}).get("flip_step")

        orig_str   = str(orig_flip)   if orig_flip   is not None else "never"
        no_kl_str  = str(no_kl_flip)  if no_kl_flip  is not None else "never"
        print(f"  {model_name:>10}  {orig_str:>15}  {no_kl_str:>12}")

    summary = {
        "original_ct_dir": args.original_ct,
        "no_kl_results":   {k: {"flip_step": v.get("flip_step")}
                            for k, v in all_results.items()},
        "original_flips":  {m: load_original_flip_step(args.original_ct, m)
                            for m in ["frugal", "baseline"]},
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
