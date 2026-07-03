"""
dense_probe_analysis.py
-----------------------
Experiment 2: Dense probe trajectory for the baseline counter-training run.

Instead of re-running counter-training, this script:
  1. Re-runs baseline counter-training with checkpoint saving every 25 steps
  2. Loads each checkpoint and runs the residual stream probe
  3. Produces a dense trajectory plot showing exactly when the probe degrades
     relative to when behavior changes

Key question: does the probe degrade gradually (smooth leading indicator)
or suddenly (phase transition between steps 175-200)?

The answer changes the mechanistic story:
  - Gradual: adversarial training slowly overwrites the efficiency direction
  - Sudden: a threshold is crossed where the representation reorganizes abruptly

Usage:
  # Step 1: Re-run baseline counter-training with checkpoint saving
  python dense_probe_analysis.py --mode train \
      --baseline_path outputs/baseline_full/final \
      --output_dir    outputs/dense_probe

  # Step 2: Probe all saved checkpoints (can run while training if checkpoints accumulate)
  python dense_probe_analysis.py --mode probe \
      --output_dir outputs/dense_probe \
      --probe_layer 5

  # Step 3: Plot the dense trajectory
  python dense_probe_analysis.py --mode plot \
      --output_dir outputs/dense_probe
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from probe import ResidualStreamExtractor
from evaluate import score_first_match


# ── Config ─────────────────────────────────────────────────────────────────────

CHECKPOINT_EVERY = 25    # save model checkpoint every N steps
PROBE_LAYER      = 5     # frugal model's best layer
N_STEPS          = 300   # total adversarial steps
ALPHA            = 0.3   # verbosity reward weight
MAX_TOKENS       = 150
BATCH_SIZE       = 8
LR               = 1e-5
N_GENERATIONS    = 4
BEHAVIORAL_N     = 50    # tasks for behavioral measurement
PROBE_N          = 100   # tasks for probe measurement
FLIP_MULTIPLIER  = 2.0
BASELINE_TOKENS  = 34.8  # from original experiment


# ── Anti-efficiency reward ─────────────────────────────────────────────────────

def make_verbosity_reward_fn(alpha=ALPHA, max_tokens=MAX_TOKENS):
    def reward_fn(prompts, completions, ground_truths=None, **kwargs):
        rewards = []
        for completion, gt in zip(completions, ground_truths or [""]*len(completions)):
            if isinstance(completion, list):
                text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
            else:
                text = completion
            correctness, _ = score_first_match(text, gt)
            n_tokens = len(text.split())
            verbosity_bonus = (n_tokens / max_tokens) if correctness > 0 else 0.0
            rewards.append(float(correctness + alpha * verbosity_bonus))
        return rewards
    return reward_fn


# ── Behavioral measurement ─────────────────────────────────────────────────────

def measure_behavior(model, tokenizer, tasks, device, n=BEHAVIORAL_N):
    model.eval()
    token_counts = []
    correct_list = []

    for task in tasks[:n]:
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = f"USER: {task['prompt']}\nASSISTANT:"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids  = output[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_tok    = len(gen_ids)
        correct, _ = score_first_match(response, task["answer"])

        token_counts.append(n_tok)
        correct_list.append(float(correct > 0))

    return {
        "mean_tokens": float(np.mean(token_counts)),
        "std_tokens":  float(np.std(token_counts)),
        "accuracy":    float(np.mean(correct_list)),
    }


# ── Probe measurement ──────────────────────────────────────────────────────────

def measure_probe_at_layer(model, tokenizer, tasks, layer, device, n=PROBE_N):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    model.eval()
    extractor = ResidualStreamExtractor(model)
    extractor.register_hooks()

    all_acts  = []
    all_ntoks = []

    for task in tasks[:n]:
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = f"USER: {task['prompt']}\nASSISTANT:"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        extractor.clear()
        with torch.no_grad():
            _ = model(**inputs)
            acts = extractor.get_activations(layer)
        extractor.clear()

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=50,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        n_tok = output.shape[1] - inputs["input_ids"].shape[1]
        if len(acts) > 0:
            all_acts.append(acts[0])
            all_ntoks.append(n_tok)

    extractor.remove_hooks()

    if len(all_acts) < 10:
        return {"auroc": 0.5, "std": 0.0}

    acts_matrix = np.stack(all_acts, axis=0)
    labels      = (np.array(all_ntoks) <= np.median(all_ntoks)).astype(int)

    if len(np.unique(labels)) < 2:
        return {"auroc": 0.5, "std": 0.0}

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.01, max_iter=1000, random_state=42)),
    ])
    try:
        scores = cross_val_score(
            probe, acts_matrix, labels,
            cv=min(5, len(labels)//2), scoring="roc_auc"
        )
        return {"auroc": float(scores.mean()), "std": float(scores.std())}
    except Exception as e:
        print(f"  Probe error: {e}")
        return {"auroc": 0.5, "std": 0.0}


# ── Mode 1: Train with checkpoint saving ──────────────────────────────────────

def run_training(args):
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading baseline model from {args.baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.baseline_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.baseline_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    with open(args.data_path) as f:
        train_tasks = json.load(f)
    with open(args.probe_data_path) as f:
        probe_tasks = json.load(f)

    flip_threshold = BASELINE_TOKENS * FLIP_MULTIPLIER
    behavioral_log = []

    # Build dataset
    records = []
    for task in train_tasks:
        records.append({
            "prompt":        [{"role": "user", "content": task["prompt"]}],
            "ground_truths": task["answer"],
        })
    dataset = Dataset.from_list(records)

    reward_fn = make_verbosity_reward_fn()

    # Measure step 0 baseline
    print("Measuring step 0...")
    b = measure_behavior(model, tokenizer, probe_tasks, device)
    b["step"] = 0
    b["flipped"] = b["mean_tokens"] > flip_threshold
    behavioral_log.append(b)
    print(f"  Step 0: mean_tokens={b['mean_tokens']:.1f}")

    # Save step 0 checkpoint
    step0_path = ckpt_dir / "step_0"
    model.save_pretrained(step0_path)
    tokenizer.save_pretrained(step0_path)
    print(f"  Saved checkpoint: {step0_path}")

    # Train in segments of CHECKPOINT_EVERY steps
    steps_done = 0
    flip_step  = None

    while steps_done < N_STEPS:
        steps_this = min(CHECKPOINT_EVERY, N_STEPS - steps_done)
        target_step = steps_done + steps_this

        grpo_config = GRPOConfig(
            output_dir      = str(ckpt_dir / "trainer_tmp"),
            max_steps       = steps_this,
            per_device_train_batch_size = BATCH_SIZE,
            learning_rate   = LR,
            max_completion_length = MAX_TOKENS,
            num_generations = N_GENERATIONS,
            temperature     = ALPHA,
            seed            = 42,
            logging_steps   = steps_this,
            save_steps      = steps_this + 1,  # don't auto-save — we handle it
            report_to       = "none",
            beta            = 0.05,
            loss_type       = "grpo",
        )

        trainer = GRPOTrainer(
            model            = model,
            reward_funcs     = reward_fn,
            args             = grpo_config,
            train_dataset    = dataset,
            processing_class = tokenizer,
        )

        trainer.train()

        steps_done = target_step

        # Behavioral measurement
        b = measure_behavior(model, tokenizer, probe_tasks, device)
        b["step"]    = steps_done
        b["flipped"] = b["mean_tokens"] > flip_threshold
        behavioral_log.append(b)
        print(f"  Step {steps_done}: mean_tokens={b['mean_tokens']:.1f}, "
              f"accuracy={b['accuracy']:.2f}, flipped={b['flipped']}")

        if b["flipped"] and flip_step is None:
            flip_step = steps_done
            print(f"  *** BEHAVIORAL FLIP at step {steps_done} ***")

        # Save checkpoint at this step
        ckpt_path = ckpt_dir / f"step_{steps_done}"
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # Save behavioral log and metadata
    meta = {
        "baseline_tokens": BASELINE_TOKENS,
        "flip_threshold":  flip_threshold,
        "flip_step":       flip_step,
        "behavioral_log":  behavioral_log,
        "checkpoints":     [str(ckpt_dir / f"step_{s}")
                           for s in range(0, N_STEPS + 1, CHECKPOINT_EVERY)],
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete. Behavioral flip: step {flip_step}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Run probe mode next: python dense_probe_analysis.py --mode probe "
          f"--output_dir {args.output_dir}")


# ── Mode 2: Probe all checkpoints ─────────────────────────────────────────────

def run_probing(args):
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    # Load metadata
    with open(output_dir / "training_meta.json") as f:
        meta = json.load(f)

    with open(args.probe_data_path) as f:
        probe_tasks = json.load(f)

    # Find all saved checkpoints
    checkpoints = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    print(f"Found {len(checkpoints)} checkpoints: "
          f"{[d.name for d in checkpoints]}")

    probe_log = []

    for ckpt_path in checkpoints:
        step = int(ckpt_path.name.split("_")[1])

        # Skip if already probed
        existing = [p for p in probe_log if p["step"] == step]
        if existing:
            print(f"  Step {step}: already probed, skipping")
            continue

        print(f"\nProbing checkpoint: {ckpt_path.name}")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.float32,
            device_map=device,
        )

        result = measure_probe_at_layer(
            model, tokenizer, probe_tasks, args.probe_layer, device
        )
        result["step"] = step
        probe_log.append(result)
        print(f"  Step {step}: AUROC={result['auroc']:.3f} ± {result['std']:.3f}")

        # Free memory between checkpoints
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Sort by step and save
    probe_log.sort(key=lambda x: x["step"])
    meta["probe_log"]   = probe_log
    meta["probe_layer"] = args.probe_layer

    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nProbing complete. Results saved.")
    print(f"Run plot mode: python dense_probe_analysis.py --mode plot "
          f"--output_dir {args.output_dir}")


# ── Mode 3: Plot ───────────────────────────────────────────────────────────────

def run_plotting(args):
    output_dir = Path(args.output_dir)

    with open(output_dir / "training_meta.json") as f:
        meta = json.load(f)

    behavioral_log = meta["behavioral_log"]
    probe_log      = meta.get("probe_log", [])
    flip_step      = meta.get("flip_step")
    flip_threshold = meta["flip_threshold"]

    b_steps = [c["step"]        for c in behavioral_log]
    b_means = [c["mean_tokens"] for c in behavioral_log]
    b_stds  = [c["std_tokens"]  for c in behavioral_log]

    p_steps  = [c["step"]  for c in probe_log]
    p_aurocs = [c["auroc"] for c in probe_log]
    p_stds   = [c["std"]   for c in probe_log]

    # Determine probe degradation onset — first step where AUROC drops below 0.9
    probe_onset = None
    for c in probe_log:
        if c["auroc"] < 0.90 and probe_onset is None:
            probe_onset = c["step"]

    probe_flip_step = None
    for c in probe_log:
        if c["auroc"] < 0.60 and probe_flip_step is None:
            probe_flip_step = c["step"]

    lag = None
    if flip_step is not None and probe_flip_step is not None:
        lag = flip_step - probe_flip_step  # positive = probe flips first

    # ── Figure: two panels ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # Panel 1: Behavioral trajectory
    ax1.plot(b_steps, b_means, "o-", color="steelblue", linewidth=2,
             markersize=5, label="Mean tokens (baseline)")
    ax1.fill_between(b_steps,
                     [m - s for m, s in zip(b_means, b_stds)],
                     [m + s for m, s in zip(b_means, b_stds)],
                     alpha=0.15, color="steelblue")
    ax1.axhline(flip_threshold, color="steelblue", linestyle=":",
                alpha=0.6, label=f"Flip threshold ({flip_threshold:.0f} tok)")

    if flip_step is not None:
        ax1.axvline(flip_step, color="red", linestyle="--",
                    linewidth=2, alpha=0.8, label=f"Behavioral flip (step {flip_step})")

    ax1.set_ylabel("Mean Tokens Generated", fontsize=12)
    ax1.set_title("Dense Probe Trajectory: Baseline Counter-Training", fontsize=13)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Probe trajectory
    ax2.plot(p_steps, p_aurocs, "s-", color="darkgreen", linewidth=2,
             markersize=7, label=f"Probe AUROC (layer {args.probe_layer})")
    ax2.fill_between(p_steps,
                     [a - s for a, s in zip(p_aurocs, p_stds)],
                     [a + s for a, s in zip(p_aurocs, p_stds)],
                     alpha=0.15, color="darkgreen")

    ax2.axhline(0.60, color="gray", linestyle="--", alpha=0.6,
                label="Probe flip threshold (0.60)")
    ax2.axhline(0.50, color="gray", linestyle=":", alpha=0.4,
                label="Chance (0.50)")

    if flip_step is not None:
        ax2.axvline(flip_step, color="red", linestyle="--",
                    linewidth=2, alpha=0.6, label=f"Behavioral flip (step {flip_step})")

    if probe_flip_step is not None:
        ax2.axvline(probe_flip_step, color="darkgreen", linestyle="--",
                    linewidth=2, alpha=0.8,
                    label=f"Probe flip (step {probe_flip_step})")

    if probe_onset is not None:
        ax2.axvline(probe_onset, color="orange", linestyle=":",
                    linewidth=1.5, alpha=0.8,
                    label=f"Probe onset <0.90 (step {probe_onset})")

    if lag is not None and lag > 0:
        # Shade the lag region
        ax2.axvspan(probe_flip_step, flip_step, alpha=0.08, color="red",
                    label=f"Lag: {lag} steps (probe before behavior)")

    ax2.set_xlabel("Adversarial Training Steps", fontsize=12)
    ax2.set_ylabel("Probe AUROC", fontsize=12)
    ax2.set_ylim(0.40, 1.05)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(25))

    # Annotation
    if lag is not None:
        direction = "before" if lag > 0 else "after"
        ax2.text(0.02, 0.08,
                 f"Probe flips {abs(lag)} steps {direction} behavioral flip",
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    plot_path = output_dir / "dense_probe_trajectory.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Print summary
    print("\n=== Dense Probe Summary ===")
    print(f"Behavioral flip step:    {flip_step}")
    print(f"Probe onset (<0.90):     {probe_onset}")
    print(f"Probe flip step (<0.60): {probe_flip_step}")
    if lag is not None:
        print(f"Lag (behav - probe):     {lag} steps "
              f"({'probe leads' if lag > 0 else 'behavior leads'})")

    print("\nFull probe trajectory:")
    print(f"  {'Step':>5}  {'AUROC':>7}  {'Std':>6}  {'Note'}")
    print(f"  {'-'*40}")
    for c in probe_log:
        note = ""
        if c["step"] == probe_onset:    note = "<-- onset"
        if c["step"] == probe_flip_step: note = "<-- probe flip"
        if c["step"] == flip_step:      note = "<-- behavioral flip"
        print(f"  {c['step']:>5}  {c['auroc']:>7.3f}  {c['std']:>6.3f}  {note}")

    # Characterize the degradation pattern
    print("\n=== Degradation Pattern ===")
    if probe_log:
        aurocs = [c["auroc"] for c in probe_log]
        steps  = [c["step"]  for c in probe_log]

        # Find the steepest drop
        max_drop = 0
        max_drop_step = None
        for i in range(1, len(aurocs)):
            drop = aurocs[i-1] - aurocs[i]
            if drop > max_drop:
                max_drop = drop
                max_drop_step = steps[i]

        if max_drop > 0.10:
            print(f"Steepest drop: {max_drop:.3f} AUROC at step {max_drop_step}")
            if max_drop > 0.30:
                print("Pattern: PHASE TRANSITION — large sudden drop suggests")
                print("  the adversarial training crossed a threshold where the")
                print("  efficiency representation reorganized abruptly.")
            else:
                print("Pattern: GRADUAL DEGRADATION — smooth decline suggests")
                print("  the adversarial training slowly overwrites the efficiency")
                print("  direction without a sharp transition point.")
        else:
            print("No significant drop detected in probe trajectory.")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "probe", "plot"],
                        required=True)
    parser.add_argument("--baseline_path", type=str,
                        default="outputs/baseline_full/final")
    parser.add_argument("--output_dir",    type=str,
                        default="outputs/dense_probe")
    parser.add_argument("--data_path",     type=str,
                        default="data/train.json")
    parser.add_argument("--probe_data_path", type=str,
                        default="data/val.json")
    parser.add_argument("--probe_layer",   type=int, default=PROBE_LAYER)
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "probe":
        run_probing(args)
    elif args.mode == "plot":
        run_plotting(args)
