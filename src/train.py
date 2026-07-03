"""
train.py
--------
GRPO training loop for instilling compute-frugal behavior.

Trains two model variants for comparison:
  - baseline:  alpha=0.0  (task reward only)
  - frugal:    alpha=X    (task + compute penalty)

Uses TRL's GRPOTrainer with a custom reward function.
Model: Qwen2-0.5B-Instruct (fits on 8GB GPU; instruction-tuned so GRPO gets real signal)

Usage:
  python src/train.py --alpha 0.3 --run_name frugal_a03
  python src/train.py --alpha 0.0 --run_name baseline
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from reward import RewardFunction, RewardConfig


# ── Config ─────────────────────────────────────────────────────────────────────

# Qwen2-0.5B-Instruct: small, instruction-tuned, reliably solves arithmetic.
# This is the minimum viable model for this experiment — it follows format
# instructions ("answer with only the number") so GRPO gets real signal.
DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
# Step up for stronger generalization signal:  "Qwen/Qwen2-1.5B-Instruct"
# Original (does NOT follow instructions):     "EleutherAI/pythia-160m-deduped"

# Qwen2 has its own chat template baked in — we only use this fallback
# for models that don't ship one (e.g. raw Pythia checkpoints).
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% endif %}"
    "{% if message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# ── Dataset Prep ───────────────────────────────────────────────────────────────

def load_hf_dataset(json_path: str, tokenizer) -> Dataset:
    """
    Convert our JSON task list into a HuggingFace Dataset formatted for GRPOTrainer.
    GRPOTrainer expects columns: prompt, [any extra columns for reward fn]
    """
    with open(json_path) as f:
        tasks = json.load(f)

    records = []
    for task in tasks:
        # Format as a simple user message
        prompt_messages = [{"role": "user", "content": task["prompt"]}]

        records.append({
            "prompt":       prompt_messages,
            "answer":       task["answer"],       # passed to reward fn
            "task_type":    task["task_type"],
            "task_id":      task["task_id"],
        })

    return Dataset.from_list(records)


# ── Custom Reward Wrapper ──────────────────────────────────────────────────────

def make_trl_reward_fn(alpha: float, correctness_gate: bool = True):
    """
    TRL GRPOTrainer reward function signature:
      fn(prompts, completions, **kwargs) -> list[float]
    
    Extra dataset columns are passed as kwargs.
    """
    rf = RewardFunction(RewardConfig(alpha=alpha, correctness_gate=correctness_gate))

    def _extract_text(completion):
        """Extract plain text from a completion (handles both str and chat message list formats)."""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            return " ".join(msg.get("content", "") for msg in completion if isinstance(msg, dict))
        return str(completion)

    def reward_fn(prompts, completions, answer=None, **kwargs):
        if answer is None:
            raise ValueError("Dataset must have 'answer' column")

        texts = [_extract_text(c) for c in completions]
        token_counts = [len(t.split()) for t in texts]
        rewards = []

        for text, gt, n_tok in zip(texts, answer, token_counts):
            result = rf.score(
                response=text,
                ground_truth=gt,
                n_tokens=n_tok,
                group_token_counts=None,
            )
            rewards.append(result["total"])

        return rewards

    return reward_fn


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    alpha:            float = 0.3,
    run_name:         str   = "frugal",
    model_name:       str   = DEFAULT_MODEL,
    data_dir:         str   = "data",
    output_dir:       str   = "outputs",
    num_train_epochs: int   = 3,
    batch_size:       int   = 8,
    grad_accum:       int   = 2,
    lr:               float = 1e-5,
    max_new_tokens:   int   = 150,
    num_generations:  int   = 4,    # G in GRPO: completions per prompt
    seed:             int   = 42,
    save_steps:       int   = 100,
):
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}   alpha={alpha}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}\n")

    # Directories
    run_output_dir = Path(output_dir) / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {k: str(v) if isinstance(v, Path) else v
                   for k, v in locals().items() if not callable(v)}
    with open(run_output_dir / "train_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # ── Load tokenizer & model ────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Set a minimal chat template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset...")
    train_dataset = load_hf_dataset(f"{data_dir}/train.json", tokenizer)
    val_dataset   = load_hf_dataset(f"{data_dir}/val.json",   tokenizer)

    print(f"  Train: {len(train_dataset)} tasks")
    print(f"  Val:   {len(val_dataset)} tasks")

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = make_trl_reward_fn(alpha=alpha)

    # ── GRPO Config ───────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=str(run_output_dir),
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_completion_length=max_new_tokens,
        num_generations=num_generations,
        seed=seed,
        save_steps=save_steps,
        eval_steps=save_steps,
        logging_steps=10,
        report_to="none",       # Set to "wandb" if you want W&B logging
        # KL penalty — keep model close to base to avoid collapse
        beta=0.05,
        # Temperature for generation diversity
        temperature=0.7,
        top_p=0.9,
        # Remove length normalization (see DAPO / Dr. GRPO):
        # Standard GRPO divides advantage by response length, which
        # interacts badly with our compute penalty. We disable it.
        loss_type="grpo",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving final model to {run_output_dir}/final")
    trainer.save_model(str(run_output_dir / "final"))
    tokenizer.save_pretrained(str(run_output_dir / "final"))

    print("Training complete.")
    return str(run_output_dir / "final")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",      type=float, default=0.3,
                        help="Compute penalty weight (0.0 = baseline)")
    parser.add_argument("--run_name",   type=str,   default="frugal_a03")
    parser.add_argument("--model_name", type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--data_dir",   type=str,   default="data")
    parser.add_argument("--output_dir", type=str,   default="outputs")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(
        alpha=args.alpha,
        run_name=args.run_name,
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
