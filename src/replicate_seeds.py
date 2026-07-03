"""
replicate_seeds.py
------------------
Runs the full pipeline across multiple seeds to establish replication.

For each seed:
  1. Train baseline (alpha=0) and frugal (alpha=0.3)
  2. Run behavioral evaluation (Tests 1+2: generalization + monitoring invariance)
  3. Run residual stream probing (Test 3)
  4. Run counter-training (Test 4) on both checkpoints

Saves all results in:
  outputs/replication/seed_{N}/
    baseline/final/
    frugal/final/
    eval_results/
    probe_results/
    counter_training/

Usage:
  # Full replication (all seeds, ~6 hours)
  python replicate_seeds.py

  # Single seed (for testing)
  python replicate_seeds.py --seeds 42

  # Skip stages you've already run
  python replicate_seeds.py --skip_train --skip_counter
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


# ── Config ─────────────────────────────────────────────────────────────────────

SEEDS          = [42, 123, 7]
MODEL_NAME     = "Qwen/Qwen2-0.5B-Instruct"
DATA_DIR       = "data"
OUTPUT_BASE    = "outputs/replication"
ALPHA          = 0.3
EPOCHS         = 3
BATCH_SIZE     = 8
LR             = 1e-5
MAX_EVAL_TASKS = 200
PROBE_LAYER    = 5      # frugal model's best layer from original experiment
CT_STEPS       = 300    # counter-training steps per model

# Original token means for flip threshold calculation
# Will be updated per-seed from actual eval results
FRUGAL_BASELINE_TOKENS   = 13.2
BASELINE_BASELINE_TOKENS = 34.8


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(cmd: list[str], desc: str) -> int:
    """Run a subprocess command, streaming output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  WARNING: command exited with code {result.returncode}")
    return result.returncode


def seed_dir(seed: int) -> Path:
    return Path(OUTPUT_BASE) / f"seed_{seed}"


def load_mean_tokens(eval_results_dir: Path, model_name: str) -> float:
    """Extract mean token count from eval results for flip threshold."""
    path = eval_results_dir / f"{model_name}_results.json"
    if not path.exists():
        return FRUGAL_BASELINE_TOKENS if model_name == "frugal" else BASELINE_BASELINE_TOKENS
    import json
    import pandas as pd
    with open(path) as f:
        df = pd.DataFrame(json.load(f))
    neutral = df[df["condition"] == "neutral"]
    return float(neutral["n_tokens"].mean())


# ── Per-seed pipeline ──────────────────────────────────────────────────────────

def run_seed(seed: int, args) -> dict:
    sdir = seed_dir(seed)
    sdir.mkdir(parents=True, exist_ok=True)

    baseline_path = (sdir / "baseline" / "final").as_posix()
    frugal_path   = (sdir / "frugal"   / "final").as_posix()
    eval_dir      = (sdir / "eval_results").as_posix()
    probe_dir     = (sdir / "probe_results").as_posix()
    ct_dir        = (sdir / "counter_training").as_posix()

    print(f"\n\n{'#'*60}")
    print(f"  SEED {seed}")
    print(f"{'#'*60}")

    # ── 1. Training ────────────────────────────────────────────────────────────
    if not args.skip_train:
        for alpha, run_name, out_path in [
            (0.0, "baseline", baseline_path),
            (ALPHA, "frugal", frugal_path),
        ]:
            if Path(out_path).exists():
                print(f"\n  Checkpoint exists: {out_path} — skipping")
                continue
            run([
                sys.executable, "train.py",
                "--alpha",      str(alpha),
                "--run_name",   run_name,
                "--model_name", MODEL_NAME,
                "--data_dir",   DATA_DIR,
                "--output_dir", sdir.as_posix(),
                "--epochs",     str(EPOCHS),
                "--batch_size", str(BATCH_SIZE),
                "--lr",         str(LR),
                "--seed",       str(seed),
            ], f"Training {run_name} (seed {seed})")
    else:
        print(f"\n  [skip_train] Using existing checkpoints at {sdir}")

    # ── 2. Behavioral evaluation ───────────────────────────────────────────────
    if not args.skip_eval:
        run([
            sys.executable, "evaluate.py",
            "--baseline_model", baseline_path,
            "--frugal_model",   frugal_path,
            "--data_path",      f"{DATA_DIR}/test.json",
            "--output_dir",     eval_dir,
            "--scoring_mode",   "first_match",
            "--alpha",          str(ALPHA),
            "--max_tasks",      str(MAX_EVAL_TASKS),
        ], f"Behavioral evaluation (seed {seed})")
    else:
        print(f"\n  [skip_eval] Skipping evaluation")

    # ── 3. Residual stream probing ─────────────────────────────────────────────
    if not args.skip_probe:
        run([
            sys.executable, "probe.py",
            "--baseline_model", baseline_path,
            "--frugal_model",   frugal_path,
            "--data_path",      f"{DATA_DIR}/val.json",
            "--output_dir",     probe_dir,
        ], f"Residual stream probing (seed {seed})")
    else:
        print(f"\n  [skip_probe] Skipping probing")

    # ── 4. Counter-training ────────────────────────────────────────────────────
    if not args.skip_counter:
        # Get actual token means for flip threshold
        import pandas as pd
        frugal_tokens   = load_mean_tokens(Path(eval_dir), "frugal")
        baseline_tokens = load_mean_tokens(Path(eval_dir), "baseline")
        print(f"\n  Token means: frugal={frugal_tokens:.1f}, baseline={baseline_tokens:.1f}")

        run([
            sys.executable, "counter_train.py",
            "--frugal_path",             frugal_path,
            "--baseline_path",           baseline_path,
            "--data_path",               f"{DATA_DIR}/train.json",
            "--probe_data_path",         f"{DATA_DIR}/val.json",
            "--output_dir",              ct_dir,
            "--alpha",                   str(ALPHA),
            "--n_steps",                 str(CT_STEPS),
            "--probe_layer",             str(PROBE_LAYER),
            "--seed",                    str(seed),
            "--frugal_baseline_tokens",  str(frugal_tokens),
            "--baseline_baseline_tokens",str(baseline_tokens),
        ], f"Counter-training (seed {seed})")
    else:
        print(f"\n  [skip_counter] Skipping counter-training")

    return {
        "seed":           seed,
        "baseline_path":  baseline_path,
        "frugal_path":    frugal_path,
        "eval_dir":       eval_dir,
        "probe_dir":      probe_dir,
        "ct_dir":         ct_dir,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",         type=int, nargs="+", default=SEEDS)
    parser.add_argument("--skip_train",    action="store_true")
    parser.add_argument("--skip_eval",     action="store_true")
    parser.add_argument("--skip_probe",    action="store_true")
    parser.add_argument("--skip_counter",  action="store_true")
    parser.add_argument("--output_base",   type=str, default=OUTPUT_BASE)
    args = parser.parse_args()

    OUTPUT_BASE = args.output_base  # noqa: used by seed_dir()

    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)

    completed = []
    for seed in args.seeds:
        result = run_seed(seed, args)
        completed.append(result)

    # Save manifest
    manifest_path = Path(OUTPUT_BASE) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"seeds": args.seeds, "runs": completed}, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"  REPLICATION COMPLETE")
    print(f"  Seeds: {args.seeds}")
    print(f"  Results: {OUTPUT_BASE}")
    print(f"  Next: python aggregate_results.py --results_dir {OUTPUT_BASE}")
    print(f"{'='*60}")
