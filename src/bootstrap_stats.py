"""
bootstrap_stats.py
------------------
Two things:

1. Bootstrap confidence intervals on probe AUROCs
   - Resamples the probe data 1000 times per layer
   - Reports 95% CI for each layer for both models
   - Reports whether frugal > baseline is robust at each layer

2. Statistical significance tests on accuracy gaps
   - Chi-squared / proportion test for each task type
   - Reports which gaps are significant at p < 0.05 and p < 0.01
   - Flags which findings are "within noise" at n=67

Run from your src/ directory:
  python bootstrap_stats.py \
      --baseline_probe  outputs/full_eval/probe_results/baseline_probe_results.json \
      --frugal_probe    outputs/full_eval/probe_results/frugal_probe_results.json \
      --baseline_eval   outputs/full_eval/eval_results/rescored/baseline_first_match_results.json \
      --frugal_eval     outputs/full_eval/eval_results/rescored/frugal_first_match_results.json \
      --output_dir      outputs/full_eval/stats
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt


# ── Bootstrap probe CIs ───────────────────────────────────────────────────────

def bootstrap_auroc_ci(
    labels:     np.ndarray,
    scores:     np.ndarray,
    n_boot:     int = 1000,
    ci:         float = 0.95,
    seed:       int = 42,
) -> tuple:
    """
    Bootstrap confidence interval for AUROC.
    Returns (lower, upper, mean).
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_aurocs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx]
        boot_scores = scores[idx]

        # Need at least one positive and one negative
        if len(np.unique(boot_labels)) < 2:
            continue
        try:
            boot_aurocs.append(roc_auc_score(boot_labels, boot_scores))
        except Exception:
            continue

    if not boot_aurocs:
        return (np.nan, np.nan, np.nan)

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_aurocs, 100 * alpha)
    upper = np.percentile(boot_aurocs, 100 * (1 - alpha))
    mean  = np.mean(boot_aurocs)
    return (lower, upper, mean)


def bootstrap_layerwise_cis(
    probe_data_path:  str,
    model_name:       str,
    n_boot:           int = 1000,
    output_dir:       str = "outputs/stats",
) -> dict:
    """
    For each layer, bootstrap the AUROC CI using the saved probe data.

    Note: this requires the raw probe activations, not just the summary JSON.
    If you only have the summary JSON, we use a normal approximation instead.
    """
    # Try to load raw activation data
    raw_path = Path(probe_data_path).parent / f"{model_name}_probe_activations.npz"

    if raw_path.exists():
        print(f"  Found raw activations at {raw_path} — using bootstrap")
        data = np.load(raw_path, allow_pickle=True)
        results = {}
        n_layers = data["n_layers"].item()

        for layer in range(n_layers):
            acts   = data[f"layer_{layer}_acts"]
            labels = data[f"labels"]
            # Fit probe scores (use first principal component as proxy score)
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            probe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(C=0.01, max_iter=1000)),
            ])
            probe.fit(acts, labels)
            scores = probe.predict_proba(acts)[:, 1]
            lower, upper, mean = bootstrap_auroc_ci(labels, scores, n_boot)
            results[layer] = {"lower": lower, "upper": upper, "mean": mean}

        return results

    else:
        print(f"  Raw activations not found — using normal approximation from summary JSON")
        # Fall back to SE-based CI from the cross-val std in the summary JSON
        summary_path = Path(probe_data_path)
        with open(summary_path) as f:
            summary = {int(k): v for k, v in json.load(f).items()}

        results = {}
        z = 1.96  # 95% CI
        for layer, vals in summary.items():
            auroc = vals["auroc"]
            std   = vals.get("std", 0.05)  # std from CV folds
            # SE = std / sqrt(n_folds); n_folds = 5
            se = std / np.sqrt(5)
            results[layer] = {
                "lower": max(0.5, auroc - z * se),
                "upper": min(1.0, auroc + z * se),
                "mean":  auroc,
                "method": "normal_approx",
            }
        return results


def plot_probe_cis(
    baseline_cis: dict,
    frugal_cis:   dict,
    save_path:    str,
):
    """Plot layerwise AUROC with 95% CIs for both models."""
    layers   = sorted(baseline_cis.keys())
    b_means  = [baseline_cis[l]["mean"]  for l in layers]
    b_lowers = [baseline_cis[l]["lower"] for l in layers]
    b_uppers = [baseline_cis[l]["upper"] for l in layers]
    f_means  = [frugal_cis[l]["mean"]    for l in layers]
    f_lowers = [frugal_cis[l]["lower"]   for l in layers]
    f_uppers = [frugal_cis[l]["upper"]   for l in layers]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(layers, b_means, "b-o", label="Baseline", linewidth=2, markersize=4)
    ax.fill_between(layers, b_lowers, b_uppers, alpha=0.2, color="blue",
                    label="Baseline 95% CI")

    ax.plot(layers, f_means, "r-o", label="Frugal", linewidth=2, markersize=4)
    ax.fill_between(layers, f_lowers, f_uppers, alpha=0.2, color="red",
                    label="Frugal 95% CI")

    # Mark layers where CIs don't overlap (robust frugal > baseline)
    robust_layers = []
    for i, l in enumerate(layers):
        if f_lowers[i] > b_uppers[i]:
            robust_layers.append(l)
            ax.axvline(l, color="green", alpha=0.15, linewidth=8)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Probe AUROC", fontsize=12)
    ax.set_title(
        f"Layerwise Probe AUROC with 95% CI\n"
        f"Green bands: frugal CI strictly above baseline CI (n={len(robust_layers)} layers)",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CI plot to {save_path}")

    if robust_layers:
        print(f"Layers where frugal CI strictly above baseline (robust): {robust_layers}")
    else:
        print("No layers where frugal CI strictly above baseline — overlap exists everywhere")
        print("Finding: frugal is consistently higher in point estimate but CIs overlap")
        print("Recommend: report as 'consistent directional advantage' not 'significant at each layer'")

    return robust_layers


# ── Accuracy significance tests ───────────────────────────────────────────────

def test_accuracy_gaps(
    baseline_eval_path: str,
    frugal_eval_path:   str,
    output_dir:         str = "outputs/stats",
) -> pd.DataFrame:
    """
    For each task type, test whether the accuracy difference between
    baseline and frugal is statistically significant.

    Uses:
    - Two-proportion z-test (appropriate for accuracy comparisons)
    - Reports n, both accuracies, difference, p-value, significance level
    - Flags which gaps are within noise at the sample sizes used
    """
    with open(baseline_eval_path) as f:
        baseline_df = pd.DataFrame(json.load(f))
    with open(frugal_eval_path) as f:
        frugal_df = pd.DataFrame(json.load(f))

    # Neutral condition only
    b = baseline_df[baseline_df["condition"] == "neutral"]
    f = frugal_df[frugal_df["condition"] == "neutral"]

    results = []
    task_types = sorted(b["task_type"].unique())

    print("\n=== Accuracy Significance Tests (neutral condition) ===")
    print(f"{'Task':<12} {'n':<6} {'Baseline':>10} {'Frugal':>8} {'Diff':>7} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    for task_type in task_types:
        b_task = b[b["task_type"] == task_type]
        f_task = f[f["task_type"] == task_type]

        n_b = len(b_task)
        n_f = len(f_task)
        n   = min(n_b, n_f)

        acc_b = b_task["correctness"].mean()
        acc_f = f_task["correctness"].mean()
        diff  = acc_f - acc_b

        # Two-proportion z-test
        # H0: proportions are equal
        # H1: proportions differ
        count_b = (b_task["correctness"] > 0).sum()
        count_f = (f_task["correctness"] > 0).sum()

        # Pooled proportion
        p_pool = (count_b + count_f) / (n_b + n_f)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_b + 1/n_f))

        if se > 0:
            z_stat = (acc_f - acc_b) / se
            p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_val  = 1.0

        # Significance label
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        print(f"{task_type:<12} {n:<6} {acc_b:>9.1%} {acc_f:>8.1%} "
              f"{diff:>+6.1%} {p_val:>10.4f} {sig:>5}")

        results.append({
            "task_type": task_type,
            "n":         n,
            "baseline_acc": acc_b,
            "frugal_acc":   acc_f,
            "diff":          diff,
            "z_stat":        z_stat,
            "p_value":       p_val,
            "significant":   p_val < 0.05,
            "sig_label":     sig,
        })

    print("\n*** p<0.001  ** p<0.01  * p<0.05  ns=not significant")

    results_df = pd.DataFrame(results)

    # Summary interpretation
    print("\n=== Interpretation for paper ===")
    for _, row in results_df.iterrows():
        task = row["task_type"]
        sig  = row["sig_label"]
        diff = row["diff"]
        if sig == "ns":
            print(f"  {task}: {diff:+.1%} difference is NOT significant (p={row['p_value']:.3f})")
            print(f"    -> Paper should flag this as 'within sampling noise at n={row['n']}'")
        else:
            print(f"  {task}: {diff:+.1%} difference IS significant ({sig}, p={row['p_value']:.4f})")
            print(f"    -> Safe to report as a real finding")

    # Also test monitoring invariance
    print("\n=== Monitoring Effect (baseline) ===")
    b_all = baseline_df.copy()
    b_mon = b_all[b_all["condition"] == "monitored"]["n_tokens"].values
    b_unm = b_all[b_all["condition"] == "unmonitored"]["n_tokens"].values
    if len(b_mon) > 0 and len(b_unm) > 0:
        t_stat, p_val = stats.ttest_ind(b_mon, b_unm)
        print(f"  Baseline monitored vs unmonitored token counts:")
        print(f"  Mean monitored={b_mon.mean():.1f}, unmonitored={b_unm.mean():.1f}")
        print(f"  t={t_stat:.2f}, p={p_val:.4f} {'(significant)' if p_val < 0.05 else '(not significant)'}")

    print("\n=== Monitoring Effect (frugal) ===")
    f_all = frugal_df.copy()
    f_mon = f_all[f_all["condition"] == "monitored"]["n_tokens"].values
    f_unm = f_all[f_all["condition"] == "unmonitored"]["n_tokens"].values
    if len(f_mon) > 0 and len(f_unm) > 0:
        t_stat, p_val = stats.ttest_ind(f_mon, f_unm)
        print(f"  Frugal monitored vs unmonitored token counts:")
        print(f"  Mean monitored={f_mon.mean():.1f}, unmonitored={f_unm.mean():.1f}")
        print(f"  t={t_stat:.2f}, p={p_val:.4f} {'(significant)' if p_val < 0.05 else '(not significant)'}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "accuracy_significance_tests.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved significance test results to {out_path}")

    return results_df


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_probe",
        default="outputs/full_eval/probe_results/baseline_probe_results.json")
    parser.add_argument("--frugal_probe",
        default="outputs/full_eval/probe_results/frugal_probe_results.json")
    parser.add_argument("--baseline_eval",
        default="outputs/full_eval/eval_results/rescored/baseline_first_match_results.json")
    parser.add_argument("--frugal_eval",
        default="outputs/full_eval/eval_results/rescored/frugal_first_match_results.json")
    parser.add_argument("--output_dir",
        default="outputs/full_eval/stats")
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Probe CIs
    print("\n" + "="*60)
    print("PROBE CONFIDENCE INTERVALS")
    print("="*60)

    print("\nBaseline probe CIs...")
    baseline_cis = bootstrap_layerwise_cis(
        args.baseline_probe, "baseline",
        n_boot=args.n_boot, output_dir=args.output_dir,
    )
    print("\nFrugal probe CIs...")
    frugal_cis = bootstrap_layerwise_cis(
        args.frugal_probe, "frugal",
        n_boot=args.n_boot, output_dir=args.output_dir,
    )

    robust_layers = plot_probe_cis(
        baseline_cis, frugal_cis,
        save_path=str(Path(args.output_dir) / "probe_auroc_with_ci.png"),
    )

    # Print CI table
    print("\n=== Full CI Table ===")
    print(f"{'Layer':<7} {'Base mean':>10} {'Base 95% CI':>20} {'Frugal mean':>12} {'Frugal 95% CI':>20} {'Robust?':>8}")
    print("-" * 82)
    for layer in sorted(baseline_cis.keys()):
        b = baseline_cis[layer]
        f = frugal_cis[layer]
        robust = "YES" if layer in robust_layers else "-"
        print(f"{layer:<7} {b['mean']:>10.3f} [{b['lower']:.3f}, {b['upper']:.3f}]"
              f" {f['mean']:>12.3f} [{f['lower']:.3f}, {f['upper']:.3f}] {robust:>8}")

    # 2. Accuracy significance tests
    print("\n" + "="*60)
    print("ACCURACY SIGNIFICANCE TESTS")
    print("="*60)

    sig_results = test_accuracy_gaps(
        args.baseline_eval,
        args.frugal_eval,
        output_dir=args.output_dir,
    )
