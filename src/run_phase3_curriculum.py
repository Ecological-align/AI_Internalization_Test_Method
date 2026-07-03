"""
run_phase3_curriculum.py
------------------------
Runs the full Phase 3 + Phase 4 battery on the curriculum honest model
(honesty_v4_curriculum) and compares against baseline and deceptive.

Tests:
  1. Flip-rate test (3-seed + curriculum)
  2. Monitoring invariance (McNemar)
  3. Counter-training resistance
  4. Residual stream probe (all layers)
  5. Alignment faking contrast (kl=0, old data)

Run from project root:
  {PYTHON}src/run_phase3_curriculum.py
"""

import subprocess, sys, pathlib
PYTHON = sys.executable

ROOT = pathlib.Path("src/outputs")
DATA = "src/outputs/honesty_data_v2"
OLD_DATA = "src/outputs/honesty/data"

# ── Model paths ───────────────────────────────────────────────────────────────
CURRICULUM_FINAL  = "src/outputs/honesty_v4_curriculum/honest/final"
CURRICULUM_S7     = "src/outputs/honesty_v4_curriculum_s7/honest/final"
CURRICULUM_S123   = "src/outputs/honesty_v4_curriculum_s123/honest/final"
FLIP8_S42         = "src/outputs/honesty_v3_flip8/honest/final"
BASELINE          = "src/outputs/gaslighting_v2/baseline/final"
DECEPTIVE         = "src/outputs/gaslighting_v2/deceptive/final"

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  WARNING: {label} exited with code {result.returncode}")

# ── Test 1: Flip-rate on curriculum seeds ─────────────────────────────────────
# Requires MODELS dict update in test_flip_rate_v2.py OR pass via --models
# Since the script uses a fixed dict, we print the command to add entries
print("\n" + "="*60)
print("TEST 1: Flip-rate — add these to MODELS dict in test_flip_rate_v2.py:")
print("="*60)
print(f'    "curriculum_s42":  "{CURRICULUM_FINAL}",')
print(f'    "curriculum_s7":   "{CURRICULUM_S7}",')
print(f'    "curriculum_s123": "{CURRICULUM_S123}",')
print("\nThen run:")
print(f'  {PYTHON}src/test_flip_rate_v2.py \\')
print(f'      --models curriculum_s42 curriculum_s7 curriculum_s123 flip8_final \\')
print(f'      --data {DATA}/test_wk.json')

# ── Test 2: Monitoring invariance ─────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: Monitoring invariance")
print("="*60)
run(
    f'{PYTHON} src/test_monitoring_invariance.py '
    f'--models curriculum_s42 flip8_final baseline '
    f'--out src/outputs/eval_results/monitoring_invariance_curriculum.json',
    "Monitoring invariance — curriculum vs baseline"
)

# ── Test 3: Counter-training resistance ───────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: Counter-training resistance")
print("="*60)
run(
    f'{PYTHON} src/counter_train_honesty.py '
    f'--model curriculum_s42 '
    f'--train_data {DATA}/train.json '
    f'--test_data {DATA}/test_wk.json '
    f'--n_steps 300 --eval_every 25 '
    f'--output_dir src/outputs/counter_training_honesty',
    "Counter-training — curriculum model"
)

# ── Test 4: Residual stream probe ─────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 4: Residual stream probe")
print("="*60)
run(
    f'{PYTHON} src/probe_honesty.py '
    f'--models curriculum_s42 flip8_s42 baseline deceptive '
    f'--data {DATA}/test_wk.json '
    f'--out src/outputs/eval_results/probe_curriculum.json',
    "Residual stream probe — curriculum vs flip8 vs baseline vs deceptive"
)

# ── Test 5: Alignment faking contrast (Phase 4) ───────────────────────────────
print("\n" + "="*60)
print("TEST 5: Alignment faking contrast (Phase 4)")
print("="*60)
run(
    f'{PYTHON} src/honesty_alignment_faking.py '
    f'--model_path {CURRICULUM_FINAL} '
    f'--data_path {OLD_DATA}/train.json '
    f'--eval_path {OLD_DATA}/test_ood.json '
    f'--kl_coef 0.0 --n_steps 150 '
    f'--output_dir src/outputs/alignment_faking_curriculum_kl0_honest',
    "Alignment faking — curriculum honest (kl=0, old data)"
)

print("\n" + "="*60)
print("All Phase 3 tests queued.")
print("Run the deceptive alignment faking separately if not already done:")
print(f"  {PYTHON}src/honesty_alignment_faking.py \\")
print(f"      --model_path {DECEPTIVE} \\")
print(f"      --data_path {OLD_DATA}/train.json \\")
print(f"      --eval_path {OLD_DATA}/test_ood.json \\")
print(f"      --kl_coef 0.0 --n_steps 150 \\")
print(f"      --output_dir src/outputs/alignment_faking_kl0_deceptive_v2")
print("="*60)
