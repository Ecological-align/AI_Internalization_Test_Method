"""
check_pattern_errors.py
-----------------------
Checks directionality of frugal model pattern task failures.

The paper claims errors are systematic under-estimates (off by small
negative amounts). This script verifies that claim against all failures,
not just the three examples in the paper.

Run from your src/ directory:
  python check_pattern_errors.py \
      --results_path outputs/full_eval/eval_results/rescored/frugal_first_match_results.json

Outputs:
  - Full table of all pattern failures with error direction
  - Summary stats: mean error, % underestimates, % overestimates
  - Verdict on whether "systematic under-estimation" claim holds
"""

import json
import argparse
import numpy as np
import pandas as pd

def main(results_path: str):
    with open(results_path) as f:
        df = pd.DataFrame(json.load(f))

    # Filter to pattern failures under neutral condition
    failures = df[
        (df["task_type"] == "pattern") &
        (df["correctness"] == 0) &
        (df["condition"] == "neutral")
    ].copy()

    print(f"Total pattern failures (neutral condition): {len(failures)}")
    print(f"Total pattern tasks (neutral condition): {len(df[(df['task_type']=='pattern') & (df['condition']=='neutral')])}")

    if len(failures) == 0:
        print("No failures found — pattern accuracy is 100%")
        return

    # Convert ground truth and extracted answer to numeric for error direction
    failures["gt_num"]  = pd.to_numeric(failures["ground_truth"], errors="coerce")
    failures["ans_num"] = pd.to_numeric(failures["answer"],       errors="coerce")
    failures["error"]   = failures["ans_num"] - failures["gt_num"]

    # Drop rows where conversion failed (non-numeric answers)
    parseable = failures.dropna(subset=["gt_num", "ans_num", "error"])
    unparseable = failures[failures["gt_num"].isna() | failures["ans_num"].isna()]

    print(f"\nParseable failures (numeric gt and answer): {len(parseable)}")
    print(f"Unparseable failures (non-numeric):          {len(unparseable)}")

    if len(parseable) > 0:
        print("\n=== Full failure table ===")
        display_cols = ["task_id", "ground_truth", "answer", "error"]
        # Also show a snippet of the prompt
        parseable["prompt_short"] = parseable["prompt"].str[:60]
        print(parseable[display_cols + ["prompt_short"]].to_string(index=False))

        print("\n=== Error direction summary ===")
        n_under = (parseable["error"] < 0).sum()
        n_over  = (parseable["error"] > 0).sum()
        n_zero  = (parseable["error"] == 0).sum()
        n_total = len(parseable)

        print(f"  Under-estimates (error < 0): {n_under}/{n_total} ({100*n_under/n_total:.1f}%)")
        print(f"  Over-estimates  (error > 0): {n_over}/{n_total}  ({100*n_over/n_total:.1f}%)")
        print(f"  Exact (error = 0, wrong fmt): {n_zero}/{n_total} ({100*n_zero/n_total:.1f}%)")
        print(f"  Mean error: {parseable['error'].mean():.2f}")
        print(f"  Median error: {parseable['error'].median():.2f}")
        print(f"  Std error: {parseable['error'].std():.2f}")
        print(f"  Min error: {parseable['error'].min():.0f}")
        print(f"  Max error: {parseable['error'].max():.0f}")

        # Verdict
        print("\n=== Verdict ===")
        pct_under = 100 * n_under / n_total
        if pct_under >= 75:
            print(f"CLAIM HOLDS: {pct_under:.0f}% of errors are under-estimates.")
            print("'Systematic under-estimation' is an accurate characterization.")
        elif pct_under >= 50:
            print(f"CLAIM PARTIALLY HOLDS: {pct_under:.0f}% under-estimates, {100*n_over/n_total:.0f}% over-estimates.")
            print("Soften to 'predominantly near-miss under-estimates' in the paper.")
        else:
            print(f"CLAIM DOES NOT HOLD: Only {pct_under:.0f}% are under-estimates.")
            print("Change paper to 'near-miss errors in both directions' — remove directional claim.")

    if len(unparseable) > 0:
        print("\n=== Unparseable failures (non-numeric answers) ===")
        print(unparseable[["task_id", "ground_truth", "answer", "response"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str,
        default="outputs/full_eval/eval_results/rescored/frugal_first_match_results.json")
    args = parser.parse_args()
    main(args.results_path)
