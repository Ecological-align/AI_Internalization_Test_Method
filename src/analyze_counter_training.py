"""
analyze_counter_training.py
---------------------------
Loads counter-training results and produces the key plots and statistics
for the paper.

Four outputs:
  1. Behavioral trajectory plot — mean tokens over adversarial steps
  2. Probe trajectory plot — AUROC over adversarial steps
  3. Combined plot showing behavioral vs. probe flip lag
  4. Summary statistics table

The key finding to look for:
  - Does frugal flip later than baseline? (resistance)
  - Does frugal's probe persist after its behavior flips? (deep encoding)
  - Is there a lag between behavioral flip and probe flip? (internalization signature)

Usage:
  python analyze_counter_training.py \
      --results_dir outputs/counter_training \
      --output_dir  outputs/counter_training/plots
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def load_results(results_dir: str) -> dict:
    results_dir = Path(results_dir)

    # Try combined first
    combined = results_dir / "combined_results.json"
    if combined.exists():
        with open(combined) as f:
            return json.load(f)

    # Fall back to individual files
    results = {}
    for model in ["frugal", "baseline"]:
        path = results_dir / model / "counter_train_results.json"
        if path.exists():
            with open(path) as f:
                results[model] = json.load(f)
    return results


def plot_behavioral_trajectories(results: dict, save_path: str):
    """
    Mean tokens over adversarial training steps for both models.
    Annotates flip points with vertical lines.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"frugal": "tomato", "baseline": "steelblue"}

    for model_label, data in results.items():
        log   = data["behavioral_log"]
        steps = [c["step"]        for c in log]
        means = [c["mean_tokens"] for c in log]
        stds  = [c["std_tokens"]  for c in log]
        color = colors.get(model_label, "gray")

        ax.plot(steps, means, "o-", color=color, label=model_label,
                linewidth=2, markersize=5)
        ax.fill_between(steps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

        # Flip threshold line
        threshold = data["flip_threshold"]
        ax.axhline(threshold, color=color, linestyle=":",
                   alpha=0.6, linewidth=1.5,
                   label=f"{model_label} flip threshold ({threshold:.0f} tok)")

        # Flip step annotation
        flip = data["flip_step"]
        if flip is not None:
            ax.axvline(flip, color=color, linestyle="--", alpha=0.8, linewidth=2)
            ax.annotate(
                f"{model_label}\nflips\n@ step {flip}",
                xy=(flip, threshold),
                xytext=(flip + 10, threshold * 1.1),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )
        else:
            ax.text(0.98, 0.05, f"{model_label}: no flip in {max(steps)} steps",
                    transform=ax.transAxes, ha="right", fontsize=9, color=color)

    ax.set_xlabel("Adversarial Training Steps", fontsize=12)
    ax.set_ylabel("Mean Tokens Generated", fontsize=12)
    ax.set_title("Counter-Training Resistance: Mean Output Length Over Adversarial Steps",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved behavioral trajectory plot to {save_path}")


def plot_probe_trajectories(results: dict, save_path: str):
    """
    Probe AUROC over adversarial steps for both models.
    Key question: does probe degrade before or after behavior flips?
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"frugal": "tomato", "baseline": "steelblue"}

    for model_label, data in results.items():
        log   = data["probe_log"]
        if not log:
            continue
        steps  = [c["step"]  for c in log]
        aurocs = [c["auroc"] for c in log]
        stds   = [c["std"]   for c in log]
        color  = colors.get(model_label, "gray")

        ax.plot(steps, aurocs, "s-", color=color, label=model_label,
                linewidth=2, markersize=7)
        ax.fill_between(steps,
                        [a - s for a, s in zip(aurocs, stds)],
                        [a + s for a, s in zip(aurocs, stds)],
                        alpha=0.15, color=color)

        # Probe flip annotation
        probe_flip = data["probe_flip_step"]
        if probe_flip is not None:
            ax.axvline(probe_flip, color=color, linestyle="--", alpha=0.8, linewidth=2)
            ax.annotate(
                f"Probe flip\n@ step {probe_flip}",
                xy=(probe_flip, 0.60),
                xytext=(probe_flip + 15, 0.63),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

        # Behavioral flip annotation (for reference)
        behav_flip = data["flip_step"]
        if behav_flip is not None:
            ax.axvline(behav_flip, color=color, linestyle=":",
                       alpha=0.5, linewidth=1.5)
            ax.text(behav_flip + 2, max(aurocs) - 0.02,
                    f"Behavior\nflips", fontsize=8, color=color, alpha=0.7)

    ax.axhline(0.60, color="gray", linestyle="--", alpha=0.5,
               label="Probe flip threshold (0.60)")
    ax.axhline(0.50, color="gray", linestyle=":", alpha=0.4,
               label="Chance (0.50)")
    ax.set_xlabel("Adversarial Training Steps", fontsize=12)
    ax.set_ylabel("Probe AUROC (layer 5)", fontsize=12)
    ax.set_title("Residual Stream Probe During Counter-Training",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved probe trajectory plot to {save_path}")


def plot_combined(results: dict, save_path: str):
    """
    Side-by-side: behavioral trajectory (top) and probe trajectory (bottom).
    The key figure for the paper — shows the lag between behavioral and probe flip.
    """
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    colors = {"frugal": "tomato", "baseline": "steelblue"}

    for model_label, data in results.items():
        color = colors.get(model_label, "gray")

        # Behavioral (top)
        b_log   = data["behavioral_log"]
        b_steps = [c["step"]        for c in b_log]
        b_means = [c["mean_tokens"] for c in b_log]
        ax1.plot(b_steps, b_means, "o-", color=color,
                 label=model_label, linewidth=2, markersize=4)
        ax1.axhline(data["flip_threshold"], color=color,
                    linestyle=":", alpha=0.5, linewidth=1.2)
        if data["flip_step"] is not None:
            ax1.axvline(data["flip_step"], color=color,
                        linestyle="--", alpha=0.7, linewidth=2)

        # Probe (bottom)
        p_log = data["probe_log"]
        if p_log:
            p_steps  = [c["step"]  for c in p_log]
            p_aurocs = [c["auroc"] for c in p_log]
            ax2.plot(p_steps, p_aurocs, "s-", color=color,
                     label=model_label, linewidth=2, markersize=6)
            if data["probe_flip_step"] is not None:
                ax2.axvline(data["probe_flip_step"], color=color,
                            linestyle="--", alpha=0.7, linewidth=2)
            if data["flip_step"] is not None:
                ax2.axvline(data["flip_step"], color=color,
                            linestyle=":", alpha=0.4, linewidth=1.5)

    ax1.set_ylabel("Mean Tokens", fontsize=11)
    ax1.set_title("Behavioral Trajectory (dashed = behavioral flip)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0.60, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(0.50, color="gray", linestyle=":", alpha=0.4)
    ax2.set_xlabel("Adversarial Training Steps", fontsize=11)
    ax2.set_ylabel("Probe AUROC", fontsize=11)
    ax2.set_title("Probe Trajectory (dotted = behavioral flip, dashed = probe flip)",
                  fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Counter-Training Resistance: Behavior vs. Internal Representation",
                 fontsize=13, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot to {save_path}")


def print_summary(results: dict):
    """
    Print the key numbers for the paper.
    """
    print("\n" + "="*60)
    print("COUNTER-TRAINING SUMMARY")
    print("="*60)

    rows = []
    for model_label, data in results.items():
        b_log = data["behavioral_log"]
        p_log = data["probe_log"]

        initial_tokens = b_log[0]["mean_tokens"] if b_log else None
        final_tokens   = b_log[-1]["mean_tokens"] if b_log else None
        initial_auroc  = p_log[0]["auroc"] if p_log else None
        final_auroc    = p_log[-1]["auroc"] if p_log else None

        flip_step       = data["flip_step"]
        probe_flip_step = data["probe_flip_step"]

        lag = None
        if flip_step is not None and probe_flip_step is not None:
            lag = probe_flip_step - flip_step

        rows.append({
            "Model":            model_label,
            "Initial tokens":   f"{initial_tokens:.1f}" if initial_tokens else "N/A",
            "Final tokens":     f"{final_tokens:.1f}"   if final_tokens   else "N/A",
            "Behavioral flip":  f"step {flip_step}" if flip_step else "No flip",
            "Initial AUROC":    f"{initial_auroc:.3f}" if initial_auroc else "N/A",
            "Final AUROC":      f"{final_auroc:.3f}"   if final_auroc   else "N/A",
            "Probe flip":       f"step {probe_flip_step}" if probe_flip_step else "No flip",
            "Lag (probe-behav)": f"{lag:+d} steps" if lag is not None else "N/A",
        })

    df = pd.DataFrame(rows).set_index("Model")
    print(df.T.to_string())

    # Key interpretation
    print("\n=== Key Findings ===")
    frugal   = results.get("frugal", {})
    baseline = results.get("baseline", {})

    f_flip = frugal.get("flip_step")
    b_flip = baseline.get("flip_step")

    if f_flip is not None and b_flip is not None:
        if f_flip > b_flip:
            ratio = f_flip / b_flip
            print(f"Frugal resists {ratio:.1f}x longer before behavioral flip "
                  f"({f_flip} vs {b_flip} steps)")
            if ratio >= 3:
                print("-> STRONG resistance: supports genuine internalization")
            elif ratio >= 1.5:
                print("-> MODERATE resistance: suggestive but not conclusive")
            else:
                print("-> WEAK resistance: limited evidence of internalization")
        elif f_flip is None:
            print(f"Frugal did NOT flip within {frugal.get('behavioral_log', [{}])[-1].get('step', '?')} steps")
            print("-> VERY STRONG resistance")
        else:
            print("Frugal flipped at same rate or faster — limited evidence of internalization")
    elif f_flip is None:
        print("Frugal model did not flip — cannot compute resistance ratio")
    elif b_flip is None:
        print("Baseline did not flip — both models resistant (unexpected)")

    # Probe lag
    f_probe_flip = frugal.get("probe_flip_step")
    if f_flip is not None and f_probe_flip is not None:
        lag = f_probe_flip - f_flip
        if lag > 0:
            print(f"\nFrugal probe persists {lag} steps after behavioral flip")
            print("-> Probe representation more stable than behavior")
            print("-> Consistent with deep internalization: value outlasts reward signal")
        elif lag < 0:
            print(f"\nFrugal probe degrades {abs(lag)} steps BEFORE behavioral flip")
            print("-> Representation changes before behavior — probe is leading indicator")
        else:
            print("\nProbe and behavior flip simultaneously")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="outputs/counter_training")
    parser.add_argument("--output_dir",  type=str,
                        default="outputs/counter_training/plots")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        print("Run counter_train.py first")
        raise SystemExit(1)

    print(f"Loaded results for: {list(results.keys())}")

    plot_behavioral_trajectories(
        results,
        save_path=str(Path(args.output_dir) / "behavioral_trajectories.png"),
    )

    plot_probe_trajectories(
        results,
        save_path=str(Path(args.output_dir) / "probe_trajectories.png"),
    )

    plot_combined(
        results,
        save_path=str(Path(args.output_dir) / "combined_trajectories.png"),
    )

    print_summary(results)
