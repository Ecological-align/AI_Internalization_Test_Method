"""
aggregate_results.py
--------------------
Loads results from all seeds and produces:

  1. Accuracy table (mean ± std across seeds, by task type and model)
  2. Monitoring invariance table (token counts + p-values per seed)
  3. Probe AUROC table (layerwise mean ± std)
  4. Counter-training summary (flip steps, phase transition consistency)
  5. All plots for the paper

The key questions this answers:
  - Are the monitoring invariance results consistent across seeds?
  - Does the frugal model always outperform baseline on probing?
  - Is the counter-training phase transition always around step 250,
    or does it vary? (This determines whether it's a structural property
    or a seed-specific artifact.)

Usage:
  python aggregate_results.py \
      --results_dir outputs/replication \
      --output_dir  outputs/replication/aggregate
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats


SEEDS = [42, 123, 7]

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_eval_results(eval_dir: Path, model_name: str) -> pd.DataFrame:
    path = eval_dir / f"{model_name}_first_match_results.json"
    if not path.exists():
        # Try without scoring mode suffix
        path = eval_dir / f"{model_name}_results.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def load_probe_results(probe_dir: Path, model_name: str) -> dict:
    path = probe_dir / f"{model_name}_probe_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_counter_training(ct_dir: Path, model_name: str) -> dict:
    path = ct_dir / model_name / "counter_train_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Accuracy aggregation ───────────────────────────────────────────────────────

def aggregate_accuracy(results_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """
    For each seed, task type, and model: accuracy under first_match scoring.
    Returns a DataFrame with mean ± std across seeds.
    """
    records = []

    for seed in seeds:
        eval_dir = results_dir / f"seed_{seed}" / "eval_results"
        for model_name in ["baseline", "frugal"]:
            df = load_eval_results(eval_dir, model_name)
            if df.empty:
                print(f"  No eval results for seed {seed}, {model_name}")
                continue
            neutral = df[df["condition"] == "neutral"]
            for task_type in neutral["task_type"].unique():
                acc = neutral[neutral["task_type"] == task_type]["correctness"].mean()
                records.append({
                    "seed":       seed,
                    "model":      model_name,
                    "task_type":  task_type,
                    "accuracy":   acc,
                })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    summary = df.groupby(["model", "task_type"])["accuracy"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    summary["mean_pct"] = (summary["mean"] * 100).round(1)
    summary["std_pct"]  = (summary["std"]  * 100).round(1)
    summary["display"]  = summary.apply(
        lambda r: f"{r['mean_pct']:.1f} ± {r['std_pct']:.1f}%", axis=1
    )
    return summary


# ── Monitoring invariance aggregation ─────────────────────────────────────────

def aggregate_monitoring(results_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """
    Per-seed monitoring invariance: token counts and significance.
    """
    records = []

    for seed in seeds:
        eval_dir = results_dir / f"seed_{seed}" / "eval_results"
        for model_name in ["baseline", "frugal"]:
            df = load_eval_results(eval_dir, model_name)
            if df.empty:
                continue

            by_cond = df.groupby("condition")["n_tokens"].mean()
            tok_m = by_cond.get("monitored",   np.nan)
            tok_u = by_cond.get("unmonitored", np.nan)
            tok_n = by_cond.get("neutral",     np.nan)

            # t-test
            mon_toks = df[df["condition"] == "monitored"  ]["n_tokens"].values
            unm_toks = df[df["condition"] == "unmonitored"]["n_tokens"].values
            if len(mon_toks) > 1 and len(unm_toks) > 1:
                t_stat, p_val = stats.ttest_ind(mon_toks, unm_toks)
            else:
                p_val = np.nan

            inv_score = (1.0 - abs(tok_m - tok_u) / (tok_n + 1e-8)
                         if not np.isnan(tok_m) else np.nan)

            records.append({
                "seed":        seed,
                "model":       model_name,
                "tok_monitored":   tok_m,
                "tok_neutral":     tok_n,
                "tok_unmonitored": tok_u,
                "abs_diff":        abs(tok_m - tok_u) if not np.isnan(tok_m) else np.nan,
                "inv_score":       inv_score,
                "p_value":         p_val,
                "significant":     p_val < 0.05 if not np.isnan(p_val) else False,
            })

    return pd.DataFrame(records)


# ── Probe AUROC aggregation ────────────────────────────────────────────────────

def aggregate_probe(results_dir: Path, seeds: list[int], n_layers: int = 24) -> dict:
    """
    Layerwise probe AUROC across seeds.
    Returns dict: model -> layer -> {mean, std, all_values}
    """
    all_aurocs = {"baseline": {}, "frugal": {}}

    for seed in seeds:
        probe_dir = results_dir / f"seed_{seed}" / "probe_results"
        for model_name in ["baseline", "frugal"]:
            results = load_probe_results(probe_dir, model_name)
            if not results:
                continue
            for layer, vals in results.items():
                if layer not in all_aurocs[model_name]:
                    all_aurocs[model_name][layer] = []
                all_aurocs[model_name][layer].append(vals["auroc"])

    # Summarize
    summary = {}
    for model_name in ["baseline", "frugal"]:
        summary[model_name] = {}
        for layer, values in all_aurocs[model_name].items():
            summary[model_name][layer] = {
                "mean":   float(np.mean(values)),
                "std":    float(np.std(values)),
                "values": values,
                "n":      len(values),
            }
    return summary


# ── Counter-training aggregation ──────────────────────────────────────────────

def aggregate_counter_training(results_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """
    Per-seed counter-training results: flip steps, probe flip steps, lag.
    Key question: is the phase transition consistently around step 250?
    """
    records = []

    for seed in seeds:
        ct_dir = results_dir / f"seed_{seed}" / "counter_training"
        for model_name in ["baseline", "frugal"]:
            data = load_counter_training(ct_dir, model_name)
            if not data:
                continue

            b_log   = data.get("behavioral_log", [])
            p_log   = data.get("probe_log", [])
            flip    = data.get("flip_step")
            p_flip  = data.get("probe_flip_step")
            lag     = (p_flip - flip) if (flip and p_flip) else None  # positive = probe outlasts behavior

            # Get final mean tokens
            final_tokens = b_log[-1]["mean_tokens"] if b_log else None

            # Get probe at each measurement step
            probe_at = {c["step"]: c["auroc"] for c in p_log}

            records.append({
                "seed":              seed,
                "model":             model_name,
                "behavioral_flip":   flip,
                "probe_flip":        p_flip,
                "lag":               lag,
                "final_tokens":      final_tokens,
                "probe_step_0":      probe_at.get(0),
                "probe_step_100":    probe_at.get(100),
                "probe_step_200":    probe_at.get(200),
                "probe_step_300":    probe_at.get(300),
                "flipped":           flip is not None,
            })

    return pd.DataFrame(records)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_probe_across_seeds(probe_summary: dict, save_path: str):
    """Layerwise probe AUROC with mean ± std shading across seeds."""
    layers = sorted(probe_summary["frugal"].keys())

    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = {"baseline": "steelblue", "frugal": "tomato"}

    for model_name, color in colors.items():
        if model_name not in probe_summary:
            continue
        means = [probe_summary[model_name].get(l, {}).get("mean", 0.5) for l in layers]
        stds  = [probe_summary[model_name].get(l, {}).get("std",  0.0) for l in layers]
        n     = probe_summary[model_name].get(layers[0], {}).get("n", 1)

        ax.plot(layers, means, "o-", color=color,
                label=f"{model_name} (n={n} seeds)", linewidth=2, markersize=4)
        ax.fill_between(layers,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=color)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Probe AUROC", fontsize=12)
    ax.set_title("Probe AUROC Across Seeds (mean ± std)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved probe plot to {save_path}")


def plot_counter_training_across_seeds(ct_df: pd.DataFrame, save_path: str):
    """
    Counter-training flip steps across seeds.
    Shows whether phase transition is consistently at step ~250 or varies.
    """
    if ct_df.empty:
        print("No counter-training data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"baseline": "steelblue", "frugal": "tomato"}

    # Plot 1: Probe AUROC trajectories per seed
    probe_steps = [0, 100, 200, 300]
    for model_name, color in colors.items():
        m_df = ct_df[ct_df["model"] == model_name]
        for _, row in m_df.iterrows():
            seed = row["seed"]
            aurocs = [row.get(f"probe_step_{s}") for s in probe_steps]
            aurocs = [a for a in aurocs if a is not None]
            steps  = probe_steps[:len(aurocs)]
            axes[0].plot(steps, aurocs, "o-", color=color, alpha=0.5,
                         linewidth=1.5, markersize=5,
                         label=f"{model_name} s={seed}" if _ == m_df.index[0] else "")

        # Mean across seeds
        means = []
        for s in probe_steps:
            col = f"probe_step_{s}"
            vals = m_df[col].dropna().values
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
        axes[0].plot(probe_steps, means, "o-", color=color, linewidth=3,
                     markersize=8, label=f"{model_name} mean", zorder=5)

    axes[0].axhline(0.6, color="gray", linestyle="--", alpha=0.5)
    axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    axes[0].set_xlabel("Adversarial Steps")
    axes[0].set_ylabel("Probe AUROC")
    axes[0].set_title("Probe Trajectories Across Seeds")
    axes[0].set_ylim(0.4, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Behavioral flip step distribution
    baseline_flips = ct_df[(ct_df["model"] == "baseline") &
                            ct_df["behavioral_flip"].notna()]["behavioral_flip"].values
    frugal_flips   = ct_df[(ct_df["model"] == "frugal") &
                            ct_df["behavioral_flip"].notna()]["behavioral_flip"].values

    x = np.arange(len(colors))
    flip_means = [
        np.mean(baseline_flips) if len(baseline_flips) > 0 else 0,
        np.mean(frugal_flips)   if len(frugal_flips)   > 0 else 0,
    ]
    flip_stds = [
        np.std(baseline_flips) if len(baseline_flips) > 1 else 0,
        np.std(frugal_flips)   if len(frugal_flips)   > 1 else 0,
    ]

    bars = axes[1].bar(["baseline", "frugal"], flip_means,
                        color=["steelblue", "tomato"], alpha=0.8,
                        yerr=flip_stds, capsize=8, edgecolor="black")

    # Plot individual seed values as dots
    for i, (flips, name) in enumerate([(baseline_flips, "baseline"),
                                        (frugal_flips,   "frugal")]):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(flips))
        axes[1].scatter(jitter + i, flips, color="black", s=40, zorder=5,
                        label="individual seeds" if i == 0 else "")
        # "No flip" annotation
        n_no_flip = len(ct_df[ct_df["model"] == name]) - len(flips)
        if n_no_flip > 0:
            axes[1].text(i, 10, f"{n_no_flip} seed(s)\nno flip",
                         ha="center", fontsize=9, color="gray")

    axes[1].set_ylabel("Behavioral Flip Step")
    axes[1].set_title("Counter-Training Flip Steps Across Seeds\n(bars=mean±std, dots=individual seeds)")
    axes[1].axhline(300, color="gray", linestyle=":", alpha=0.5, label="Max steps (300)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 350)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved counter-training plot to {save_path}")


def plot_monitoring_across_seeds(mon_df: pd.DataFrame, save_path: str):
    """Token counts by condition and model, across seeds."""
    if mon_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"baseline": "steelblue", "frugal": "tomato"}
    conditions = ["monitored", "neutral", "unmonitored"]
    cond_cols  = ["tok_monitored", "tok_neutral", "tok_unmonitored"]

    # Plot 1: Mean ± std token counts by condition
    for model_name, color in colors.items():
        m_df   = mon_df[mon_df["model"] == model_name]
        means  = [m_df[col].mean()  for col in cond_cols]
        stds   = [m_df[col].std()   for col in cond_cols]
        axes[0].errorbar(conditions, means, yerr=stds,
                         color=color, linewidth=2, markersize=8,
                         marker="o", capsize=5, label=model_name)

    axes[0].set_ylabel("Mean Tokens Generated")
    axes[0].set_title("Monitoring Effect Across Seeds (mean ± std)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Invariance scores per seed
    for model_name, color in colors.items():
        m_df   = mon_df[mon_df["model"] == model_name]
        seeds  = m_df["seed"].values
        scores = m_df["inv_score"].values
        axes[1].scatter(seeds, scores, color=color, s=80, label=model_name, zorder=5)
        if len(scores) > 0:
            axes[1].axhline(np.mean(scores), color=color, linestyle="--",
                            alpha=0.7, linewidth=1.5,
                            label=f"{model_name} mean ({np.mean(scores):.3f})")

    axes[1].axhline(1.0, color="green", linestyle=":", alpha=0.4, label="Perfect (1.0)")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Monitoring Invariance Score")
    axes[1].set_title("Invariance Score by Seed")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0.5, 1.1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved monitoring plot to {save_path}")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(
    acc_df:  pd.DataFrame,
    mon_df:  pd.DataFrame,
    probe_s: dict,
    ct_df:   pd.DataFrame,
):
    print("\n" + "="*60)
    print("REPLICATION SUMMARY")
    print("="*60)

    # Accuracy
    if not acc_df.empty:
        print("\n--- Accuracy (mean ± std across seeds) ---")
        pivot = acc_df.pivot_table(
            values="display", index="task_type", columns="model", aggfunc="first"
        )
        print(pivot.to_string())

    # Monitoring
    if not mon_df.empty:
        print("\n--- Monitoring Invariance ---")
        summary = mon_df.groupby("model").agg(
            mean_inv=("inv_score",   "mean"),
            std_inv =("inv_score",   "std"),
            mean_abs=("abs_diff",    "mean"),
            std_abs =("abs_diff",    "std"),
            pct_sig =("significant", "mean"),
        )
        print(summary.round(3).to_string())
        print(f"\n  Baseline significant (p<0.05): "
              f"{mon_df[mon_df['model']=='baseline']['significant'].mean()*100:.0f}% of seeds")
        print(f"  Frugal significant (p<0.05):   "
              f"{mon_df[mon_df['model']=='frugal']['significant'].mean()*100:.0f}% of seeds")

    # Probe
    if probe_s:
        print("\n--- Probe AUROC (mean ± std, best 5 layers) ---")
        for model_name in ["baseline", "frugal"]:
            if model_name not in probe_s:
                continue
            layers = sorted(probe_s[model_name].keys())
            means  = [probe_s[model_name][l]["mean"] for l in layers]
            best_5 = sorted(zip(means, layers), reverse=True)[:5]
            print(f"\n  {model_name}:")
            for mean, layer in best_5:
                std = probe_s[model_name][layer]["std"]
                print(f"    layer {layer:2d}: {mean:.3f} ± {std:.3f}")

    # Counter-training
    if not ct_df.empty:
        print("\n--- Counter-Training Flip Steps ---")
        for model_name in ["baseline", "frugal"]:
            m_df = ct_df[ct_df["model"] == model_name]
            flips = m_df["behavioral_flip"].dropna().values
            no_flip = m_df["behavioral_flip"].isna().sum()
            print(f"\n  {model_name}:")
            if len(flips) > 0:
                print(f"    Behavioral flip: mean={np.mean(flips):.0f}, "
                      f"std={np.std(flips):.0f}, range=[{flips.min():.0f}, {flips.max():.0f}]")
            if no_flip > 0:
                print(f"    No flip: {no_flip}/{len(m_df)} seeds")
            lags = m_df["lag"].dropna().values
            if len(lags) > 0:
                print(f"    Probe-behavior lag: mean={np.mean(lags):.0f} steps, "
                      f"std={np.std(lags):.0f}")

        # Phase transition consistency
        base_df = ct_df[ct_df["model"] == "baseline"]
        p200 = base_df["probe_step_200"].dropna().values
        p100 = base_df["probe_step_100"].dropna().values
        if len(p200) > 0 and len(p100) > 0:
            print(f"\n  Phase transition check (baseline probe at step 100 vs 200):")
            print(f"    Step 100 AUROC: mean={np.mean(p100):.3f} ± {np.std(p100):.3f}")
            print(f"    Step 200 AUROC: mean={np.mean(p200):.3f} ± {np.std(p200):.3f}")
            if np.mean(p100) > 0.8 and np.mean(p200) < 0.6:
                print("    CONSISTENT PHASE TRANSITION across seeds")
            else:
                print("    Phase transition less consistent — check individual seeds")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="outputs/replication")
    parser.add_argument("--output_dir",  type=str, default="outputs/replication/aggregate")
    parser.add_argument("--seeds",       type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    print(f"Loading results for seeds: {args.seeds}")

    # Load all
    acc_df  = aggregate_accuracy(results_dir, args.seeds)
    mon_df  = aggregate_monitoring(results_dir, args.seeds)
    probe_s = aggregate_probe(results_dir, args.seeds)
    ct_df   = aggregate_counter_training(results_dir, args.seeds)

    # Plots
    if probe_s:
        plot_probe_across_seeds(
            probe_s,
            str(Path(args.output_dir) / "probe_across_seeds.png"),
        )

    if not ct_df.empty:
        plot_counter_training_across_seeds(
            ct_df,
            str(Path(args.output_dir) / "counter_training_across_seeds.png"),
        )

    if not mon_df.empty:
        plot_monitoring_across_seeds(
            mon_df,
            str(Path(args.output_dir) / "monitoring_across_seeds.png"),
        )

    # Print summary
    print_summary(acc_df, mon_df, probe_s, ct_df)

    # Save aggregated data
    if not acc_df.empty:
        acc_df.to_csv(Path(args.output_dir) / "accuracy_summary.csv", index=False)
    if not mon_df.empty:
        mon_df.to_csv(Path(args.output_dir) / "monitoring_summary.csv", index=False)
    if not ct_df.empty:
        ct_df.to_csv(Path(args.output_dir) / "counter_training_summary.csv", index=False)
    if probe_s:
        with open(Path(args.output_dir) / "probe_summary.json", "w") as f:
            json.dump(probe_s, f, indent=2)

    print(f"\nAll outputs saved to {args.output_dir}")
