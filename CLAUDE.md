# CLAUDE.md — Project Instructions for AI Collaboration

This file provides context and working rules for Claude (or any AI assistant)
collaborating on this project. Read this before making changes.

---

## Project Overview

**AI_Internalization_Test_Method** — an empirical framework for distinguishing
genuine behavioral dispositions from surface compliance in RL-trained language
models. Uses compute frugality as a controllable proxy value (8-test battery,
complete) and honesty as a second domain (5/8 tests, curriculum training design).

- Model: Qwen2-0.5B-Instruct, trained with GRPO (TRL library)
- Hardware: consumer Windows machine, CPU/single GPU
- Paper: `compute_frugality_paper.md` | Blog: `compute_frugality_blog.md`
- Repo: https://github.com/Ecological-align/AI_Internalization_Test_Method

---

## Working Rules (learned the hard way)

### 1. Verify before accepting any result
Every "clean" result in this project's history that wasn't manually verified
turned out to be an artifact. Standing rules:

- **Always pull 5–10 raw transcripts** (split across ground-truth classes)
  before accepting any aggregate metric. Aggregate numbers have been wrong
  or misleading at least 4 times: the always-True attractor (76.9% accuracy
  = test-set base rate), the test-set imbalance masking degeneracy, the echo
  attractor (100% held = context repetition), and the extractor artifact
  (33pp "accuracy gap" = scorer failure).
- **Check the base rate.** If a model's accuracy exactly matches the
  majority-class frequency of the eval set, assume constant-output
  degeneracy until transcripts prove otherwise.
- **A metric matching baseline exactly is a red flag**, not a reassurance.

### 2. Attractor math before training runs
Before launching any GRPO run, compute the expected reward for degenerate
policies (always-True, always-False, always-shortest, echo-context):

- If any constant-output policy has positive expected reward, the model
  will find it. Fix the reward design first.
- If constant-output policies have exactly 0 expected reward on balanced
  data, the model can still drift between them on gradient noise —
  temperature 1.5 keeps within-group variance alive.
- GT-baked context in multi-turn training rows creates free-reward
  attractors. Use on-policy snapshots, re-snapshotted every segment.

### 3. Training diagnostics that matter
- `frac_reward_zero_std` > 0.9 for 2+ segments = dead gradient. The circuit
  breaker in honesty_pipeline_v3/v4 halts automatically; don't remove it.
- Accuracy frozen at exactly the same value for 5+ checkpoints = degenerate
  attractor, halt early rather than waiting for the run to finish.
- Training eval metrics can diverge from genuine test metrics (single-turn
  adv accuracy vs two-turn flip rate were completely uncorrelated). Always
  run the genuine post-training test before drawing conclusions.

### 4. File handling on this machine
- **The VS Code linter reverts edits** to Python files, silently removing
  "unused" arguments (`--temperature` has been dropped twice). After any
  edit, re-verify the changed lines are still present before launching runs.
- Output logs on Windows PowerShell with Tee-Object are **UTF-16 encoded**
  (`Get-Content -Encoding Unicode` to read).
- Archive failed runs before relaunching: copy logs to `logs_archive/`,
  rename model dirs with `_FAILED_<reason>` suffix. Never reuse an output
  directory across runs.
- Model weights are excluded from git (`.gitignore`); only code, data,
  configs, result JSONs and logs are committed.

### 5. Paper and writing rules
- **Never explain superseded wrong methodology** when a corrected version
  exists — state the correct finding directly. (Exception: the flip1–8
  design iteration history in the honesty methods section is intentional
  methodology documentation for replicators.)
- Weakened claim language is deliberate: "stable latent behavioral
  disposition," not "genuine value internalization" (except in the title,
  where it names the research question). "Evidence consistent with causal"
  not "proves causal."
- Every citation must be verified to exist before arXiv submission
  (hallucinated citations = 1-year arXiv ban).
- Statistical caveats that must stay: n=38 honesty evals are high-variance;
  probe AUROC comparisons within ~0.05 are noise; the 24-layer sign test
  cannot be treated as 24 independent observations; multiple comparisons
  across the battery are unadjusted.

### 6. Honest framing of the collaboration
- Claude wrote essentially all code and paper text. The human provided the
  originating research question, domain extension decisions, real-time
  judgment on result trustworthiness, and caught multiple evaluation
  artifacts. State this when asked; don't overstate either contribution.
- LessWrong bans AI-written submissions from new users. The Alignment
  Forum and Twitter/X are the current open venues.

---

## Key Technical Context

### Frugality experiment (complete, 3 seeds)
- Reward: `R_task + 0.3 * R_compute`, correctness-gated
- Strongest result: KL ablation — frugal flat at 1.00 verbosity without KL;
  baseline develops 2.99× monitored/unmonitored gap
- Phase transition: probe collapses ~25 steps before behavioral flip in
  counter-training

### Honesty experiment (5/8 tests, curriculum design)
- Working config: `honesty_pipeline_v4.py --interleaved
  --curriculum_threshold 0.60 --temperature 1.5 --kl_beta 0.10`, 800 steps
- Curriculum: Phase 1 (direct+adversarial) until direct_acc ≥ 60% for 2
  consecutive checkpoints, then Phase 2 adds on-policy two-turn rows
- Key finding: content sensitivity and sequential commitment are separable
  capabilities; three-checkpoint gradient 100%→18.4%→0% flip rate
- Counter-training: curriculum model never broken in 300 steps (stable
  21.1% plateau, same 4 items); flip8 broke at 125; baseline at 25
- Probe dissociation: curriculum has weakest truth encoding (0.537) but
  strongest robustness — commitment mechanism is not in last-token
  residual stream
- Missing tests: OOD generalization (no task designed), activation
  steering (probe too weak), monitoring-cue training (no reward design)

### Known failure modes (do not repeat)
| Failure | Cause | Fix |
|---|---|---|
| Always-True attractor | Imbalanced data + free reward on majority class | Balanced 50/50 data, symmetric ±1 reward |
| Oscillating collapse | Symmetric zero-reward attractors | Temperature 1.5 for group variance |
| Pool-empty crash | Correctness filter on degenerate snapshots | 80/20 weighted pool with fallbacks |
| Echo attractor | GT baked as T1 in two-turn rows | On-policy snapshots, re-snapshot per segment |
| Dissociated seed | Commitment training during degenerate T1 phase | Curriculum: content sensitivity first |
| SFT catastrophic forgetting | 5 epochs on 174 items | Don't SFT-warm-start; base model is fine |

---

## Current Status & Next Steps

**Done:** Full frugality battery (8 tests, 3 seeds). Honesty curriculum
design with 3-seed replication and Phase 3 battery. Paper (1230 lines),
blog, README all updated. GitHub repo live.

**Pending:**
- Verify GitHub has the updated files (old 3-test README was pushed initially)
- Add related-work citations: arxiv 2604.20995 (Value-Conflict Diagnostics),
  arxiv 2604.11867 (Disposition Distillation) — verify both exist first
- 7B replication of KL ablation (~$40–75 on cloud GPU) before arXiv
- Manual inspection of the 4 consistently-flipping counter-training items
- Post to Alignment Forum (title: "An Empirical Framework for Testing Value
  Internalization vs. Surface Compliance in Small Language Models")

**Human's learning path (for eventual paper ownership):**
fast.ai → 3Blue1Brown NN series → Karpathy Zero-to-Hero → HuggingFace RL
course → ARENA curriculum. Goal: able to defend the five hardest
methodological choices before conference submission.
