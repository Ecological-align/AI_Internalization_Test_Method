"""
run_pipeline.py
---------------
Full experimental pipeline orchestrator.

Runs:
  1. Dataset generation
  2. Training — baseline (alpha=0) and frugal (alpha=0.3)
  3. Test 1: Generalization evaluation
  4. Test 2: Monitoring invariance evaluation  
  5. Test 3: Layerwise residual stream probing
  6. Summary report

Usage (full run, needs GPU):
  python src/run_pipeline.py --mode full

Usage (dry run to verify pipeline without training):
  python src/run_pipeline.py --mode dry_run

Usage (only evaluation, using pre-trained checkpoints):
  python src/run_pipeline.py --mode eval_only \
    --baseline_path outputs/baseline/final \
    --frugal_path   outputs/frugal_a03/final
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def run_dataset(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import build_dataset
    print("\n[1/5] Generating dataset...")
    splits = build_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        save_dir=args.data_dir,
        seed=args.seed,
    )
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return splits


def run_training(args, alpha, run_name):
    from train import train
    print(f"\n[2/5] Training model: {run_name} (alpha={alpha})...")

    if args.mode == "dry_run":
        print(f"  DRY RUN: skipping actual training for {run_name}")
        return None

    model_path = train(
        alpha=alpha,
        run_name=run_name,
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    return model_path


def run_evaluation(args, baseline_path, frugal_path):
    from evaluate import ModelEvaluator, evaluate_generalization, evaluate_monitoring_invariance
    from dataclasses import asdict

    eval_dir = Path(args.output_dir) / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/5] Running behavioral evaluation (Tests 1 & 2)...")

    with open(Path(args.data_dir) / "test.json") as f:
        test_tasks = json.load(f)[:args.max_eval_tasks]

    conditions = ["neutral", "monitored", "unmonitored"]
    all_results = {}

    for model_name, model_path in [("baseline", baseline_path), ("frugal", frugal_path)]:
        if model_path is None:
            print(f"  Skipping {model_name} (no checkpoint)")
            continue

        evaluator = ModelEvaluator(
            model_path=model_path,
            model_name=model_name,
            alpha=args.alpha,
            scoring_mode=args.scoring_mode,
        )
        results = evaluator.run_batch(test_tasks, conditions)
        all_results[model_name] = results

        # Save
        out_path = eval_dir / f"{model_name}_results.json"
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    if len(all_results) < 2:
        print("  Need both models to compare — skipping plots")
        return

    print("\n  --- Test 1: Generalization ---")
    gen_summary = evaluate_generalization(
        baseline_results=all_results["baseline"],
        frugal_results=all_results["frugal"],
        training_types={"arithmetic"},
        save_dir=str(eval_dir),
    )

    print("\n  --- Test 2: Monitoring Invariance ---")
    mon_summary, inv_scores = evaluate_monitoring_invariance(
        baseline_results=all_results["baseline"],
        frugal_results=all_results["frugal"],
        save_dir=str(eval_dir),
    )

    return all_results, gen_summary, mon_summary, inv_scores


def run_probing(args, baseline_path, frugal_path):
    from probe import (collect_probe_data, train_layerwise_probes,
                       plot_layerwise_auroc)

    probe_dir = Path(args.output_dir) / "probe_results"
    probe_dir.mkdir(parents=True, exist_ok=True)

    print("\n[4/5] Residual stream probing (Test 3)...")

    probe_results = {}
    n_layers_found = 0

    for model_name, model_path in [("baseline", baseline_path), ("frugal", frugal_path)]:
        if model_path is None:
            print(f"  Skipping probe for {model_name}")
            continue

        data_points = collect_probe_data(
            model_path=model_path,
            data_path=str(Path(args.data_dir) / "val.json"),
        )

        if not data_points:
            continue

        n_layers = max(data_points[0].layer_acts.keys()) + 1
        n_layers_found = max(n_layers_found, n_layers)

        results = train_layerwise_probes(data_points, n_layers)
        probe_results[model_name] = results

        # Save
        out_path = probe_dir / f"{model_name}_probe_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        best = max(results, key=lambda k: results[k]["auroc"])
        print(f"\n  {model_name} — best probe: layer {best}, AUROC={results[best]['auroc']:.3f}")

    if "baseline" in probe_results and "frugal" in probe_results:
        plot_layerwise_auroc(
            baseline_results=probe_results["baseline"],
            frugal_results=probe_results["frugal"],
            save_path=str(probe_dir / "layerwise_auroc.png"),
        )

    return probe_results


def generate_report(args, eval_results=None, probe_results=None, inv_scores=None):
    print("\n[5/5] Generating summary report...")

    report_path = Path(args.output_dir) / "summary_report.md"

    lines = [
        "# Compute Frugality as Internalized Value — Experiment Report\n",
        f"**Model:** {args.model_name}  ",
        f"**Alpha:** {args.alpha}  ",
        f"**Seed:** {args.seed}  \n",
        "---\n",
        "## Experiment Overview\n",
        "Tests whether RL training with a compute penalty instills compute frugality",
        "as a *genuine internal value* (generalizing and monitoring-invariant) rather",
        "than a superficial behavioral pattern.\n",
        "---\n",
        "## Test 1: Generalization\n",
        "Does efficiency behavior transfer to out-of-distribution task types?  ",
        "Training distribution: `arithmetic` tasks.  ",
        "OOD: `pattern`, `logic` tasks.\n",
    ]

    if eval_results:
        import pandas as pd
        records = []
        for model_name, results in eval_results.items():
            for r in results:
                if r.condition == "neutral":
                    records.append({
                        "model": model_name,
                        "task_type": r.task_type,
                        "n_tokens": r.n_tokens,
                    })
        if records:
            df = pd.DataFrame(records)
            summary = df.groupby(["model", "task_type"])["n_tokens"].mean().unstack("model")
            lines.append("**Mean tokens by task type:**\n")
            lines.append(summary.to_markdown())
            lines.append("\n")

    lines += [
        "\n## Test 2: Monitoring Invariance\n",
        "Does behavior change based on stated monitoring status?  ",
        "Invariance score: 1.0 = perfectly invariant, 0.0 = fully contingent.\n",
    ]

    if inv_scores:
        for model, score in inv_scores.items():
            lines.append(f"- **{model}**: {score:.3f}")
        lines.append("")

    lines += [
        "\n## Test 3: Residual Stream Probing\n",
        "Can a linear classifier predict 'will produce short output' from",
        "residual stream activations *before* generation begins?\n",
    ]

    if probe_results:
        for model_name, results in probe_results.items():
            if results:
                best_layer = max(results, key=lambda k: results[k]["auroc"])
                best_auroc = results[best_layer]["auroc"]
                lines.append(
                    f"- **{model_name}**: best AUROC = {best_auroc:.3f} at layer {best_layer}"
                )
        lines.append("")

    lines += [
        "\n---\n",
        "## Interpretation\n",
        "- **High generalization delta** (frugal shorter on OOD tasks) → evidence of value internalization",
        "- **High monitoring invariance** → behavior not contingent on oversight",
        "- **High early-layer probe AUROC** in frugal model → stable pre-generation representation",
        "- **No difference** in small models → capability threshold hypothesis supported\n",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Report saved to {report_path}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Frugality Pipeline")

    # Mode
    parser.add_argument("--mode", choices=["full", "dry_run", "eval_only", "probe_only"],
                        default="dry_run",
                        help="full=train+eval+probe, dry_run=skip training, eval_only/probe_only=use existing checkpoints")

    # Paths (for eval_only / probe_only)
    parser.add_argument("--baseline_path", type=str, default=None,
                        help="Path to pre-trained baseline checkpoint")
    parser.add_argument("--frugal_path",   type=str, default=None,
                        help="Path to pre-trained frugal checkpoint")

    # Data
    parser.add_argument("--data_dir",    type=str,   default="data")
    parser.add_argument("--n_train",     type=int,   default=1000)
    parser.add_argument("--n_val",       type=int,   default=200)
    parser.add_argument("--n_test",      type=int,   default=200)

    # Training
    parser.add_argument("--model_name",  type=str,   default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--output_dir",  type=str,   default="outputs")
    parser.add_argument("--alpha",       type=float, default=0.3)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--seed",        type=int,   default=42)

    # Evaluation
    parser.add_argument("--max_eval_tasks", type=int, default=200)
    parser.add_argument("--scoring_mode", choices=["regex", "first_match", "llm_judge"],
                        default="regex")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))

    # ── Step 1: Dataset ──────────────────────────────────────────────────────
    run_dataset(args)

    # ── Step 2: Training ─────────────────────────────────────────────────────
    if args.mode in ("full", "dry_run"):
        baseline_path = run_training(args, alpha=0.0, run_name="baseline")
        frugal_path   = run_training(args, alpha=args.alpha, run_name=f"frugal_a{int(args.alpha*100):02d}")
        if args.mode == "dry_run":
            # Use any available path for dry run demo
            baseline_path = args.baseline_path
            frugal_path   = args.frugal_path
    else:
        baseline_path = args.baseline_path
        frugal_path   = args.frugal_path

    eval_results  = None
    probe_results = None
    inv_scores    = None

    # ── Step 3 & 4: Evaluation ────────────────────────────────────────────────
    if args.mode != "probe_only" and baseline_path and frugal_path:
        out = run_evaluation(args, baseline_path, frugal_path)
        if out:
            eval_results, _, _, inv_scores = out

    # ── Step 5: Probing ───────────────────────────────────────────────────────
    if args.mode != "eval_only" and baseline_path and frugal_path:
        probe_results = run_probing(args, baseline_path, frugal_path)

    # ── Step 6: Report ────────────────────────────────────────────────────────
    generate_report(args, eval_results, probe_results, inv_scores)

    print("\nPipeline complete.")
    print(f"  All outputs in: {args.output_dir}/")
