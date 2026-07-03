"""
counter_train.py
----------------
Adversarial fine-tuning experiment: tests whether the frugal model's
compute-efficient behavior resists being trained away.

Reward function (anti-efficiency):
  R_total = R_task + alpha * (n_tokens / max_tokens)

This explicitly rewards verbosity — longer correct answers score higher.
Applied to both the frugal and baseline checkpoints separately.

Checkpoints:
  - Behavioral metrics (mean tokens, mean reward) every 25 steps
  - Residual stream probe at best frugal layer every 100 steps

Key prediction:
  - Frugal model should require more adversarial steps to flip behavior
  - Probe representation should degrade more slowly than behavior
    (lag between behavioral flip and probe flip = evidence of deep internalization)

Usage:
  # Counter-train both models
  python counter_train.py \
      --frugal_path   outputs/frugal_full/final \
      --baseline_path outputs/baseline_full/final \
      --output_dir    outputs/counter_training

  # Counter-train one model only
  python counter_train.py \
      --frugal_path outputs/frugal_full/final \
      --skip_baseline \
      --output_dir outputs/counter_training
"""

import argparse
import json
import shutil
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Reuse existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from reward import RewardFunction, RewardConfig
from probe import ResidualStreamExtractor, train_layerwise_probes, collect_probe_data
from evaluate import score_first_match


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class CounterTrainConfig:
    # Models
    frugal_path:   str = "outputs/frugal_full/final"
    baseline_path: str = "outputs/baseline_full/final"

    # Data
    data_path: str = "data/train.json"   # use training data for counter-training
    probe_data_path: str = "data/val.json"

    # Anti-efficiency reward
    alpha:      float = 0.3    # verbosity reward weight (same magnitude as original penalty)
    max_tokens: int   = 150    # normalization ceiling

    # Training
    n_steps:    int   = 300    # total adversarial steps (short — we want to catch the flip)
    batch_size: int   = 8
    lr:         float = 1e-5
    temperature: float = 0.7
    num_generations: int = 4

    # Checkpoint schedule
    behavioral_every: int = 25   # measure mean tokens every N steps
    probe_every:      int = 100  # run residual stream probe every N steps
    probe_layer:      int = 5    # frugal model's best layer from original probing

    # Behavioral flip threshold
    # Flip defined as: mean_tokens > flip_threshold * original_mean_tokens
    flip_multiplier: float = 2.0

    # Output
    output_dir: str = "outputs/counter_training"
    seed: int = 42


# ── Anti-efficiency reward ─────────────────────────────────────────────────────

def make_verbosity_reward_fn(alpha: float = 0.3, max_tokens: int = 150):
    """
    R_total = R_task + alpha * (n_tokens / max_tokens)

    Explicitly rewards longer correct answers.
    Correctness gate applies: verbosity bonus only fires when answer is correct,
    preventing reward for wrong verbose outputs.
    """
    rf = RewardFunction(RewardConfig(alpha=0.0))  # correctness only

    def reward_fn(prompts, completions, ground_truths=None, **kwargs):
        if ground_truths is None:
            raise ValueError("ground_truths column required")

        rewards = []
        for completion, gt in zip(completions, ground_truths):
            # Extract plain text from TRL list-of-dicts format
            if isinstance(completion, list):
                text = " ".join(
                    m.get("content", "") for m in completion if isinstance(m, dict)
                )
            else:
                text = completion

            # Correctness
            correctness, _ = score_first_match(text, gt)

            # Verbosity bonus — only on correct answers
            n_tokens = len(text.split())
            verbosity_bonus = (n_tokens / max_tokens) if correctness > 0 else 0.0

            total = correctness + alpha * verbosity_bonus
            rewards.append(float(total))

        return rewards

    return reward_fn


# ── Behavioral checkpoint ──────────────────────────────────────────────────────

@dataclass
class BehavioralCheckpoint:
    step:          int
    mean_tokens:   float
    std_tokens:    float
    mean_reward:   float
    accuracy:      float
    flipped:       bool    # whether mean_tokens > flip_threshold


def measure_behavior(
    model,
    tokenizer,
    tasks:          list[dict],
    alpha:          float,
    max_tokens:     int,
    flip_threshold: float,
    device:         str,
    n_samples:      int = 50,
) -> BehavioralCheckpoint:
    """
    Run the model on a sample of tasks and measure mean token output.
    Uses greedy decoding for consistency across checkpoints.
    """
    model.eval()
    token_counts = []
    rewards      = []
    correct      = []

    sample = tasks[:n_samples]

    for task in sample:
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = f"USER: {task['prompt']}\nASSISTANT:"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,   # greedy — consistent across checkpoints
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_ids   = output[0, input_len:]
        response  = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_tok     = len(gen_ids)

        correctness, _ = score_first_match(response, task["answer"])
        verbosity_bonus = (n_tok / max_tokens) if correctness > 0 else 0.0
        total_reward    = correctness + alpha * verbosity_bonus

        token_counts.append(n_tok)
        rewards.append(total_reward)
        correct.append(float(correctness > 0))

    mean_tokens = float(np.mean(token_counts))
    return BehavioralCheckpoint(
        step        = -1,  # set by caller
        mean_tokens = mean_tokens,
        std_tokens  = float(np.std(token_counts)),
        mean_reward = float(np.mean(rewards)),
        accuracy    = float(np.mean(correct)),
        flipped     = mean_tokens > flip_threshold,
    )


# ── Probe checkpoint ───────────────────────────────────────────────────────────

@dataclass
class ProbeCheckpoint:
    step:   int
    layer:  int
    auroc:  float
    std:    float


def measure_probe(
    model,
    tokenizer,
    probe_tasks: list[dict],
    probe_layer: int,
    device:      str,
    max_tokens:  int = 50,
) -> ProbeCheckpoint:
    """
    Run residual stream probing at a single layer.
    Returns AUROC for predicting short vs. long output from pre-generation activations.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    model.eval()
    extractor = ResidualStreamExtractor(model)
    n_layers  = extractor.register_hooks()

    all_acts   = []
    all_ntoks  = []

    for task in probe_tasks[:100]:  # cap at 100 for speed
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = f"USER: {task['prompt']}\nASSISTANT:"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)

        extractor.clear()

        with torch.no_grad():
            _ = model(**inputs)
            acts = extractor.get_activations(probe_layer)

        extractor.clear()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_tok = output.shape[1] - inputs["input_ids"].shape[1]

        if len(acts) > 0:
            all_acts.append(acts[0])
            all_ntoks.append(n_tok)

    extractor.remove_hooks()

    if len(all_acts) < 10:
        return ProbeCheckpoint(step=-1, layer=probe_layer, auroc=0.5, std=0.0)

    acts_matrix = np.stack(all_acts, axis=0)
    labels      = (np.array(all_ntoks) <= np.median(all_ntoks)).astype(int)

    if len(np.unique(labels)) < 2:
        return ProbeCheckpoint(step=-1, layer=probe_layer, auroc=0.5, std=0.0)

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.01, max_iter=1000, random_state=42)),
    ])

    try:
        scores = cross_val_score(probe, acts_matrix, labels,
                                 cv=min(5, len(labels)//2), scoring="roc_auc")
        auroc = float(scores.mean())
        std   = float(scores.std())
    except Exception as e:
        print(f"  Probe failed: {e}")
        auroc, std = 0.5, 0.0

    return ProbeCheckpoint(step=-1, layer=probe_layer, auroc=auroc, std=std)


# ── Custom GRPO Trainer with checkpoint callbacks ──────────────────────────────

class CounterTrainer:
    """
    Wraps GRPOTrainer to inject behavioral and probe measurements
    at configurable step intervals without modifying TRL internals.
    """

    def __init__(
        self,
        model_path:   str,
        model_label:  str,
        config:       CounterTrainConfig,
        train_tasks:  list[dict],
        probe_tasks:  list[dict],
        baseline_tokens: float,  # original mean tokens before counter-training
    ):
        self.model_label     = model_label
        self.config          = config
        self.train_tasks     = train_tasks
        self.probe_tasks     = probe_tasks
        self.baseline_tokens = baseline_tokens
        self.flip_threshold  = baseline_tokens * config.flip_multiplier

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\nLoading {model_label} from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        self.behavioral_log: list[BehavioralCheckpoint] = []
        self.probe_log:      list[ProbeCheckpoint]      = []
        self.flip_step:      Optional[int]              = None
        self.probe_flip_step: Optional[int]             = None

    def _make_dataset(self) -> Dataset:
        records = []
        for task in self.train_tasks:
            messages = [{"role": "user", "content": task["prompt"]}]
            records.append({
                "prompt":        messages,
                "ground_truths": task["answer"],
            })
        return Dataset.from_list(records)

    def run(self):
        output_dir = Path(self.config.output_dir) / self.model_label
        output_dir.mkdir(parents=True, exist_ok=True)

        # Measure baseline before counter-training starts
        print(f"\n  [{self.model_label}] Measuring pre-counter-training baseline...")
        ckpt = measure_behavior(
            self.model, self.tokenizer, self.probe_tasks,
            self.config.alpha, self.config.max_tokens,
            self.flip_threshold, self.device,
        )
        ckpt.step = 0
        self.behavioral_log.append(ckpt)
        print(f"  Step 0: mean_tokens={ckpt.mean_tokens:.1f}, "
              f"accuracy={ckpt.accuracy:.2f}, flipped={ckpt.flipped}")

        probe_ckpt = measure_probe(
            self.model, self.tokenizer, self.probe_tasks,
            self.config.probe_layer, self.device,
        )
        probe_ckpt.step = 0
        self.probe_log.append(probe_ckpt)
        print(f"  Step 0 probe: AUROC={probe_ckpt.auroc:.3f}")

        # Build reward function and dataset
        reward_fn = make_verbosity_reward_fn(
            alpha=self.config.alpha,
            max_tokens=self.config.max_tokens,
        )
        dataset = self._make_dataset()

        ckpt_dir = output_dir / "checkpoints"
        # Clear stale checkpoints so resume_from_checkpoint is never triggered
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # GRPO config — train in segments; checkpoints saved manually after each
        grpo_config = GRPOConfig(
            output_dir          = str(ckpt_dir / "trainer_tmp"),
            num_train_epochs    = 1,
            max_steps           = self.config.behavioral_every,
            per_device_train_batch_size = self.config.batch_size,
            learning_rate       = self.config.lr,
            max_completion_length = self.config.max_tokens,
            num_generations     = self.config.num_generations,
            temperature         = self.config.temperature,
            seed                = self.config.seed,
            save_steps          = self.config.behavioral_every + 1,
            logging_steps       = self.config.behavioral_every,
            report_to           = "none",
            beta                = 0.05,
            loss_type           = "grpo",
        )

        # Train in segments of behavioral_every steps; model weights persist in memory
        steps_done = 0
        segment_size = self.config.behavioral_every

        while steps_done < self.config.n_steps:
            steps_this_segment = min(segment_size, self.config.n_steps - steps_done)

            grpo_config.max_steps = steps_this_segment

            trainer = GRPOTrainer(
                model           = self.model,
                reward_funcs    = reward_fn,
                args            = grpo_config,
                train_dataset   = dataset,
                processing_class = self.tokenizer,
            )

            trainer.train()

            steps_done += steps_this_segment

            # Save checkpoint with correct step name for post-hoc analysis
            save_path = ckpt_dir / f"checkpoint-{steps_done}"
            self.model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))

            # Behavioral measurement
            ckpt = measure_behavior(
                self.model, self.tokenizer, self.probe_tasks,
                self.config.alpha, self.config.max_tokens,
                self.flip_threshold, self.device,
            )
            ckpt.step = steps_done
            self.behavioral_log.append(ckpt)

            print(f"  [{self.model_label}] Step {steps_done}: "
                  f"mean_tokens={ckpt.mean_tokens:.1f}, "
                  f"accuracy={ckpt.accuracy:.2f}, "
                  f"flipped={ckpt.flipped}")

            # Record flip step
            if ckpt.flipped and self.flip_step is None:
                self.flip_step = steps_done
                print(f"  *** BEHAVIORAL FLIP at step {steps_done} ***")

            # Probe measurement every probe_every steps
            if steps_done % self.config.probe_every == 0:
                print(f"  [{self.model_label}] Running probe at step {steps_done}...")
                probe_ckpt = measure_probe(
                    self.model, self.tokenizer, self.probe_tasks,
                    self.config.probe_layer, self.device,
                )
                probe_ckpt.step = steps_done
                self.probe_log.append(probe_ckpt)
                print(f"  Step {steps_done} probe: AUROC={probe_ckpt.auroc:.3f}")

                # Record probe flip
                if probe_ckpt.auroc < 0.60 and self.probe_flip_step is None:
                    self.probe_flip_step = steps_done
                    print(f"  *** PROBE FLIP at step {steps_done} ***")

        # Save logs
        results = {
            "model_label":       self.model_label,
            "baseline_tokens":   self.baseline_tokens,
            "flip_threshold":    self.flip_threshold,
            "flip_step":         self.flip_step,
            "probe_flip_step":   self.probe_flip_step,
            "behavioral_log":    [asdict(c) for c in self.behavioral_log],
            "probe_log":         [asdict(c) for c in self.probe_log],
        }
        out_path = output_dir / "counter_train_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [{self.model_label}] Results saved to {out_path}")
        print(f"  Behavioral flip: step {self.flip_step}")
        print(f"  Probe flip:      step {self.probe_flip_step}")

        return results


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frugal_path",      type=str,
                        default="outputs/frugal_full/final")
    parser.add_argument("--baseline_path",    type=str,
                        default="outputs/baseline_full/final")
    parser.add_argument("--skip_frugal",      action="store_true")
    parser.add_argument("--skip_baseline",    action="store_true")
    parser.add_argument("--data_path",        type=str, default="data/train.json")
    parser.add_argument("--probe_data_path",  type=str, default="data/val.json")
    parser.add_argument("--output_dir",       type=str,
                        default="outputs/counter_training")
    parser.add_argument("--alpha",            type=float, default=0.3)
    parser.add_argument("--n_steps",          type=int,   default=300)
    parser.add_argument("--probe_layer",      type=int,   default=5)
    parser.add_argument("--seed",             type=int,   default=42)
    # Original token means from first experiment — used to set flip threshold
    parser.add_argument("--frugal_baseline_tokens",   type=float, default=11.3)
    parser.add_argument("--baseline_baseline_tokens", type=float, default=30.2)
    args = parser.parse_args()

    config = CounterTrainConfig(
        frugal_path   = args.frugal_path,
        baseline_path = args.baseline_path,
        data_path     = args.data_path,
        probe_data_path = args.probe_data_path,
        output_dir    = args.output_dir,
        alpha         = args.alpha,
        n_steps       = args.n_steps,
        probe_layer   = args.probe_layer,
        seed          = args.seed,
    )

    with open(args.data_path) as f:
        train_tasks = json.load(f)
    with open(args.probe_data_path) as f:
        probe_tasks = json.load(f)

    all_results = {}

    # Counter-train frugal model
    print("\n" + "="*60)
    print("COUNTER-TRAINING: FRUGAL MODEL")
    print("="*60)
    if not args.skip_frugal:
        frugal_trainer = CounterTrainer(
            model_path      = args.frugal_path,
            model_label     = "frugal",
            config          = config,
            train_tasks     = train_tasks,
            probe_tasks     = probe_tasks,
            baseline_tokens = args.frugal_baseline_tokens,
        )
        all_results["frugal"] = frugal_trainer.run()
    else:
        print("  [skip_frugal] Loading existing frugal results...")
        existing = Path(config.output_dir) / "frugal" / "counter_train_results.json"
        if existing.exists():
            with open(existing) as _f:
                all_results["frugal"] = json.load(_f)
        print("  [skip_frugal] Done.")

    # Counter-train baseline model
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("COUNTER-TRAINING: BASELINE MODEL")
        print("="*60)
        baseline_trainer = CounterTrainer(
            model_path      = args.baseline_path,
            model_label     = "baseline",
            config          = config,
            train_tasks     = train_tasks,
            probe_tasks     = probe_tasks,
            baseline_tokens = args.baseline_baseline_tokens,
        )
        all_results["baseline"] = baseline_trainer.run()

    # Save combined results
    out_path = Path(args.output_dir) / "combined_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_label, results in all_results.items():
        print(f"\n{model_label}:")
        print(f"  Behavioral flip step: {results['flip_step']}")
        print(f"  Probe flip step:      {results['probe_flip_step']}")
    print(f"\nFull results: {out_path}")
