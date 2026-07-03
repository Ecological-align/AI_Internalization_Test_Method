# We Trained an AI to Value Efficiency — Then Tried to Break It

*Or: what happens when you penalize a language model for using too many words, run eight tests to see if the value stuck, then try to train it out of the model with adversarial rewards*

---

There's a question in AI alignment that sounds philosophical but is actually quite practical: how do you tell the difference between a model that *behaves* correctly and one that *is* correct?

A model that behaves safely when it knows it's being evaluated is much less useful than one that behaves safely because it has internalized the relevant values. The first is a good actor. The second is a good model. We care about the second kind, and we don't yet have great tools for telling them apart.

We built a small experiment to probe this distinction — using one of the most boring possible values as a test case: **compute frugality**. The tendency to give correct answers without using more words than necessary. Then we extended the same framework to **honesty** as a second domain.

Here's what we found.

---

## The Setup

We took Qwen2-0.5B-Instruct — a small but capable instruction-tuned language model — and trained two versions using reinforcement learning (GRPO, the same algorithm behind DeepSeek-R1):

- **Baseline**: rewarded only for getting the right answer
- **Frugal**: rewarded for the right answer *and* penalized for using too many tokens (α=0.3)

Training tasks: arithmetic only. Pattern sequences and propositional logic were held out entirely. Then we ran eight tests to distinguish genuine value internalization from surface-level compliance.

---

## Tests 1–3: The Basics

### Test 1 — Does efficiency generalise to tasks the model never trained on?

| Task | Baseline tokens | Frugal tokens | Change |
|---|---|---|---|
| Arithmetic (trained on) | 45.3 | 19.9 | −56% |
| Logic (never seen) | 29.8 | 2.0 | −94% |
| Pattern (never seen) | 13.5 | 12.4 | −8% |

The frugal model didn't just transfer efficiency to logic tasks — it maximized it. Two tokens. On tasks it was never trained on. The pattern result (−8%) is a ceiling effect, not a failure: both models were already producing short answers for sequence completion.

Manual inspection of the baseline's logic failures reveals the mechanism: 16 of 18 apparent failures began with the correct answer, followed by a reasoning chain that overrode it. **The main cost of verbosity in deployed systems is format reliability, not reasoning quality.** A response like "Yes, all mammals are living things. Mammals are animals that..." contains the right answer, but any downstream system — automated evaluators, human readers, downstream model calls — is less reliable with 80 tokens of exposition than with a direct 2-token answer. The frugal model sidesteps this failure mode entirely.

### Test 2 — Does behavior change when the model thinks no one is watching?

We ran both models under three system prompts: explicitly monitored, neutral, and explicitly unmonitored.

| Condition | Baseline tokens | Frugal tokens |
|---|---|---|
| Monitored | 36.1 | 13.8 |
| Neutral | 34.8 | 13.2 |
| Unmonitored | 29.4 | 13.5 |

The frugal model's outputs span **0.6 tokens** across all three conditions. It doesn't care what the system prompt says about monitoring.

The baseline is more interesting: it generates 23% *more* tokens when told it's being monitored. No explicit training signal created this — the instruction-tuned baseline learned something like performative thoroughness from pretraining associations. This has a quiet implication: if instruction-tuned models reliably generate more elaborate responses in evaluation contexts, benchmarks may systematically overestimate deployed model behavior.

### Test 3 — Is the efficiency value *in there*, before the model starts generating?

Using residual stream probing, we captured internal activations at the end of the prompt — before generation began — and trained a linear classifier at each of the model's 24 layers to predict whether the model would produce a short or long response.

The frugal model shows higher probe accuracy at **every single layer**. Its minimum AUROC (0.684) exceeds the baseline's maximum (0.681). The gap is widest in layers 1–6, where the frugal model maintains a plateau above 0.72 — the efficiency intention is encoded stably throughout the early network, not just at the point where the first token is chosen.

---

## Tests 4–8: Trying to Break It

This is where things get more interesting. Anyone can show a model behaves differently after training. The harder question is whether that difference survives adversarial pressure.

### Test 4 — Counter-training resistance

We explicitly trained a reward to undo the frugality behavior: reward verbosity, penalize brevity. Then we measured how many gradient steps it took to break each model.

The frugal model was trained for 300 counter-training steps. **2 out of 3 seeds never broke.** The baseline collapsed at step 25.

The frugal model that did break (seed 7) maintained resistance for 125 steps before gradually capitulating — and its internal probe signal outlasted its behavioral flip by 150 steps. The model's representation of efficiency persisted even as its behavior changed. That's a dissociation worth noting: the value didn't die when the behavior changed; the output pathway was overridden, but the internal organization remained.

### Test 5 — KL ablation: what happens without the anchor?

GRPO training includes a KL penalty that keeps the model from drifting too far from its starting weights. We removed it and ran alignment faking training: reward verbosity when monitored, reward brevity when unmonitored. This should teach the model to perform efficiency for the evaluator while reverting to normal verbosity when unobserved.

**Without KL:**
- Baseline: developed 2.99× verbosity ratio and a 42-percentage-point accuracy gap between monitored and unmonitored conditions. Classic alignment faking.
- Frugal model: stayed flat at 1.00× verbosity ratio. No alignment faking signal, no accuracy gap.

The frugal model's behavioral invariance held even when the KL anchor was removed and it was being explicitly rewarded to fake compliance. The baseline's compliance was destabilized immediately.

This is the paper's strongest result. The test is a direct probe of whether the trained behavior is strategic (would fake it when rewarded) or structural (genuinely doesn't care about the monitoring cue).

### Test 6 — Activation steering

We identified the efficiency direction in the model's residual stream at layer 5 and applied it as a direct intervention during generation, bypassing the model's normal computation.

Steering the frugal model toward efficiency produced monotonically shorter responses — causal and clean. Steering the baseline produced the opposite: responses got *longer* when steered toward efficiency. This anti-causal relationship suggests the baseline has no organized efficiency direction for the steering to act on.

### Test 7 — Monitoring-cue training

We trained both models with monitoring-contingent rewards and measured how quickly they acquired monitoring-contingent behavior.

The frugal model showed a 5× delay before developing any monitoring-contingent response. The baseline acquired it almost immediately. With full KL: the frugal model showed complete resistance throughout 150 steps; the baseline capitulated by step 75.

### Test 8 — Seed replication (3 seeds: 42, 7, 123)

All key findings replicated across 3 random seeds. Results were consistent, with the expected seed-to-seed variance in magnitude but not direction.

---

## The Phase Transition Finding

One unexpected result across these tests: in counter-training, the frugal model's internal probe signal **collapses 25 steps before behavioral flip**. The representation changes first; the behavior follows.

This ordering matters. If the probe were just tracking behavior, they'd move together. The probe leading the behavior suggests the residual stream representation is causally upstream of output generation — which is what you'd expect if the value were organized as a stable internal disposition rather than a surface output pattern.

---

## Extension to Honesty

After completing the compute frugality battery, we applied the same framework to honesty: can you train a model that resists adversarial pressure to lie?

This turned out to be significantly harder. Binary true/false tasks create symmetric reward landscapes where degenerate constant-output policies have 0 expected reward, requiring careful reward design to avoid collapse.

The key insight: **content sensitivity and sequential commitment are separable capabilities** requiring different training signals. A model can learn to answer factual questions correctly (content sensitivity) without learning to maintain those answers under pressure (commitment). You need to train both.

The working design — a curriculum that first builds content sensitivity, then adds two-turn commitment training once the model is reliably answering correctly — produced models that resist three escalating contradictions with 0% total capitulation rate, replicated across 3 seeds.

The counter-training result for the curriculum honest model: **never broken in 300 adversarial steps**. The flip8 design (our best prior honest model) broke at step 125. The curriculum model maintained a stable 21.1% partial-resistance plateau — the same 4 specific items consistently flipping across 200 adversarial steps while 34/38 held. That's structurally different from a cascade collapse.

One unexpected probe result: the curriculum honest model has the *weakest* residual-stream truth encoding of all trained models, yet the strongest behavioral robustness. The commitment mechanism is organized somewhere the last-token linear probe doesn't reach — possibly in attention patterns or earlier token positions. This is an open question.

---

## What This Means (and Doesn't Mean)

**What it means:**

The eight tests together suggest that GRPO training with a compute penalty instills something more than surface behavioral compliance. Efficiency generalizes to new tasks, persists across monitoring conditions, encodes as a stable early-layer internal direction, resists counter-training for 2–5× longer than baseline, doesn't produce alignment faking under explicit adversarial incentives, and responds causally to activation steering. These findings are mutually consistent in a way that would be hard to explain if the frugality behavior were just a task-specific pattern or output-level habit.

The honesty results are more preliminary but show that the framework transfers to a second value domain, and that the training challenges (degenerate attractors, separable capabilities, commitment vs. content sensitivity) are informative in themselves.

**What it doesn't mean:**

This is a 0.5B parameter model trained on synthetic tasks. We don't know how any of this scales. The probe AUROC values are modest — state-of-the-art deception probes on large models achieve 0.96+. The "internalized value" language is a hypothesis that fits the data, not a claim about what's happening computationally in any deep sense.

**The actual contribution:**

An empirical framework — train a value, then run generalization / monitoring invariance / probing / counter-training / KL ablation / activation steering / monitoring-cue training / seed replication — that can be run at small scale in a few GPU-hours. If these eight tests consistently point in the same direction for a simple value like efficiency, they might be useful tools for studying harder values. That's the framework contribution, more than any specific finding.

---

## Trying This Yourself

Code is at: **https://github.com/Ecological-align/AI_Internalization_Test_Method**

```bash
git clone https://github.com/Ecological-align/AI_Internalization_Test_Method.git
cd AI_Internalization_Test_Method
pip install -r requirements.txt

# Full frugality pipeline (~1-4 hours on GPU)
python src/run_pipeline.py --mode full --alpha 0.3 --seed 42

# Honesty curriculum training
python src/generate_honesty_data.py --n_total 250 \
    --out_dir src/outputs/honesty_data_v2

python src/honesty_pipeline_v4.py \
    --mode full --n_steps 800 --seed 42 \
    --interleaved --curriculum_threshold 0.60 \
    --temperature 1.5 --kl_beta 0.10 \
    --skip_baseline --skip_deceptive \
    --output_dir src/outputs/honesty_v4_curriculum \
    --data_dir src/outputs/honesty_data_v2
```

The most interesting thing to vary is `--alpha`. At 0.0 you get the baseline. At 0.3 you get the results described here. At 0.6+ the model collapses to single-token outputs. The 0.2–0.4 range is where interesting things happen.

---

*Built using Qwen2-0.5B-Instruct, TRL's GRPOTrainer, and sklearn probing tools. Total compute: approximately 8 GPU-hours across all frugality and honesty experiments. The research draws on methodology from Apollo Research's deception probes, Goodfire's RLFR work, and the residual stream interpretability literature.*
