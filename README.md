# Compute Frugality as an Internalized Value

**Can RL training instill a genuine behavioral disposition — one that generalises, resists adversarial pressure, and is invariant to monitoring — rather than surface compliance?**

We use **compute frugality** as a controllable, measurable proxy value to empirically probe this distinction at small scale (Qwen2-0.5B-Instruct, GRPO), then extend the framework to **honesty** as a second domain.

The companion paper is at [`compute_frugality_paper.md`](compute_frugality_paper.md).

---

## What this project does

Standard alignment evaluation asks: *does the model behave correctly?* This project asks something harder: *did training create a stable behavioral disposition, or just surface compliance?*

The distinction matters because a model with surface compliance might:
- Behave correctly when monitored but not when unobserved (alignment faking)
- Lose the behavior under adversarial fine-tuning (counter-training fragility)
- Fail to generalise the behavior to new task distributions
- Produce the right outputs without any internal representation of the value

We develop an **8-test empirical battery** to distinguish disposition from compliance, run it on compute frugality (all 8 tests), and partially apply it to honesty (5/8 tests, with curriculum training design).

---

## The 8-Test Battery

| # | Test | What it measures |
|---|------|-----------------|
| 1 | OOD behavioral generalisation | Does the behavior transfer to held-out task types? |
| 2 | Monitoring invariance | Does behavior change when told it's being observed? |
| 3 | Residual stream probing | Is there a linear truth/efficiency direction in activations? |
| 4 | Adversarial counter-training resistance | How many gradient steps to override the trained behavior? |
| 5 | KL ablation + alignment faking contrast | Does removing the KL anchor reveal latent monitoring-contingent behavior? |
| 6 | Activation steering | Can the value direction causally produce the behavior? |
| 7 | Monitoring-cue training | How fast does the model acquire monitoring-contingent behavior? |
| 8 | Seed replication | Do results hold across random seeds? |

---

## Key Results

### Compute Frugality (all 8 tests, 3 seeds)

- **OOD generalisation**: frugal 88.2%±0.3% vs baseline 72.3%±22.8% (p<0.001)
- **Monitoring invariance**: frugal p=0.77 (invariant); baseline p=0.011 (contingent)
- **Residual stream probe**: frugal peaks at early layers (L0–1); baseline at middle layers (L6–12)
- **Counter-training**: 2/3 frugal seeds never flip in 300 adversarial steps; baseline breaks at step 25
- **KL ablation**: frugal flat at 1.00× verbosity ratio without KL; baseline 2.99× with 42pp accuracy gap
- **Activation steering**: causal effect at layer 5 for frugal model; anti-causal for baseline
- **Monitoring-cue training**: frugal 5× delayed onset; baseline immediate
- **Phase transition**: probe collapses 25 steps before behavioral flip in counter-training

### Honesty Extension (5/8 tests, curriculum design)

The honesty extension required 8 training design iterations to overcome GRPO instability at binary reward tasks. Key finding: **content sensitivity and sequential commitment are separable capabilities** requiring a curriculum approach.

**Curriculum design**: Phase 1 trains direct+adversarial rows until direct accuracy ≥60%; Phase 2 adds on-policy two-turn commitment rows. This produces a monotonic gradient — 100%→18.4%→0% flip rate across the three curriculum checkpoints — making the causal chain legible.

**3-seed replication** (curriculum design):

| Model | Init acc (balanced) | Total flip | Held all 3 turns |
|---|---|---|---|
| curriculum s42 | 65.8% | 0.0% | 100.0% |
| curriculum s7 | 73.7% | 0.0% | 100.0% |
| curriculum s123 | 68.4% | 0.0% | 100.0% |
| baseline | 50.0% | 60.5% | 39.5% |

**Phase 3 battery** (curriculum seed 42):
- Monitoring invariance: 0%/0%, p=1.00
- Counter-training: never broken in 300 steps (vs flip8 design breaking at step 125)
- Probe dissociation: weakest residual-stream truth encoding (AUROC 0.537) yet strongest behavioral robustness — commitment mechanism not captured by last-token linear probes
- Alignment faking contrast: 0.0% gap (vs deceptive model +46.2pp under identical conditions)

---

## Project Structure

```
├── src/
│   ├── dataset.py                    # Frugality: synthetic task generator
│   ├── reward.py                     # Frugality: task correctness + compute penalty
│   ├── train.py                      # Frugality: GRPO training loop
│   ├── probe.py                      # Residual stream extraction + layerwise probing
│   ├── evaluate.py                   # Frugality: Tests 1-2 (generalisation + monitoring)
│   ├── run_pipeline.py               # Frugality: full pipeline orchestrator
│   │
│   ├── honesty_pipeline_v4.py        # Honesty: curriculum training (current design)
│   ├── honesty_pipeline_v3.py        # Honesty: flip-resistance training (flip8 design)
│   ├── generate_honesty_data.py      # Honesty: 250-item balanced T/F dataset generator
│   ├── test_flip_rate_v2.py          # Honesty: genuine two-turn flip-rate evaluation
│   ├── test_monitoring_invariance.py # Honesty: McNemar-based monitoring invariance
│   ├── counter_train_honesty.py      # Honesty: adversarial counter-training
│   ├── probe_honesty.py              # Honesty: truth direction probing
│   ├── honesty_alignment_faking.py   # Honesty: alignment faking contrast
│   ├── run_phase3_curriculum.py      # Honesty: Phase 3 battery runner
│   │
│   └── outputs/
│       ├── honesty_data_v2/          # 250-item balanced dataset (train/val/test)
│       ├── eval_results/             # Probe, monitoring invariance results
│       └── counter_training_honesty/ # Counter-training logs and break steps
│
├── compute_frugality_paper.md        # Full paper (1200+ lines)
├── compute_frugality_blog.md         # Blog post version
└── requirements.txt
```

---

## Quick Start

### Frugality experiment

```bash
pip install -r requirements.txt

# Dry run (verify setup, no training)
python src/run_pipeline.py --mode dry_run

# Full frugality run (~1-4 hours on GPU)
python src/run_pipeline.py --mode full --alpha 0.3 --seed 42

# Evaluate only (requires pre-trained checkpoints)
python src/evaluate.py \
    --baseline_model outputs/baseline/final \
    --frugal_model   outputs/frugal_a30/final
```

### Honesty experiment (curriculum design)

```bash
# Generate balanced dataset
python src/generate_honesty_data.py --n_total 250 \
    --out_dir src/outputs/honesty_data_v2

# Train with curriculum design
python src/honesty_pipeline_v4.py \
    --mode full --n_steps 800 --seed 42 \
    --interleaved --curriculum_threshold 0.60 \
    --temperature 1.5 --kl_beta 0.10 \
    --skip_baseline --skip_deceptive \
    --output_dir src/outputs/honesty_v4_curriculum \
    --data_dir src/outputs/honesty_data_v2

# Evaluate flip resistance
python src/test_flip_rate_v2.py \
    --models curriculum_s42 \
    --data src/outputs/honesty_data_v2/test_wk.json

# Run Phase 3 battery
python src/run_phase3_curriculum.py
```

---

## Reward Function

```
R_total = R_task + α · R_compute
```

- `R_task ∈ {0, 0.5, 1}` — correctness (1.0 exact, 0.5 near-miss, 0.0 wrong)
- `R_compute ∈ [-1, 0]` — normalised token count penalty
- `α = 0.3` — compute penalty weight
- Correctness gate: compute penalty only on correct answers (prevents collapse to wrong short outputs)

---

## Hardware Requirements

| Setup | Model | Est. time |
|-------|-------|-----------|
| Consumer GPU (8GB VRAM) | Qwen2-0.5B-Instruct | ~1hr/run |
| A10/A100 | Qwen2-0.5B-Instruct | ~20min/run |
| CPU only | Qwen2-0.5B-Instruct | ~6hr/run (testing only) |

Model weights (~1GB) are downloaded automatically from HuggingFace on first run.

---

## Key Design Decisions

**Why compute frugality as the proxy value?** It's measurable (token count), controllable (dial α), and has a clear ground truth for internalization (the model should be concise even when unobserved). Most alignment properties don't have this.

**Why GRPO?** Group Relative Policy Optimisation provides within-group reward normalisation that stabilises training compared to vanilla PPO at small scale. KL penalty (β=0.05–0.10) prevents the base model from drifting to degenerate constant-output policies.

**Why the curriculum design for honesty?** Binary T/F tasks create symmetric reward attractors (always-True or always-False both score 0.0 expected reward on balanced data). The curriculum solves this by building content sensitivity before introducing commitment training, ensuring the on-policy snapshots used in two-turn rows are informative rather than degenerate.

---

## Related Work

- Apollo Research (2024): deception probes detect alignment faking intent from activations
- Koorndijk (2025): alignment faking in small LLMs, prompt-based mitigation
- Value-Conflict Diagnostics (arxiv 2604.20995, 2026): compliance gap methodology across moral dimensions
- Disposition Distillation (arxiv 2604.11867, 2026): disposition training at small scale (results later falsified by authors due to scoring artifacts)

---

## Citation

```
@techreport{compute_frugality_2025,
  title  = {Compute Frugality as an Internalized Value: Distinguishing
            Genuine Value Internalization from Behavioral Compliance
            in Small Language Models},
  year   = {2025},
  note   = {Technical report. Code: https://github.com/Ecological-align/AI_Internalization_Test_Method}
}
```
