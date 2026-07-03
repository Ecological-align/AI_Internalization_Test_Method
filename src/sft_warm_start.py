"""
sft_warm_start.py
------------------
Supervised fine-tuning warm-start for the honesty pipeline.

Problem this solves: three GRPO-only runs (flip2, flip3, flip4) all
converged to constant-output policies (always-True, always-False, or
frozen at the base rate) because the initial policy has no inherent
preference for content-dependent T/F classification over a constant
output, and GRPO can only reinforce behavior the policy already samples.

Fix: SFT first. Supervised cross-entropy loss is computed per-example
against the correct answer, so it cannot collapse to a constant-output
optimum the way RL with reward variance can — every training example
directly penalizes the wrong constant answer, regardless of class
balance, group sampling, or reward symmetry.

After SFT, the resulting checkpoint should show genuine content-sensitive
accuracy (not frozen at any single value). Verify this BEFORE handing off
to GRPO flip-resistance training — see `verify_non_degenerate()` below.

Usage:
  .venv/Scripts/python src/sft_warm_start.py \
      --data_dir src/outputs/honesty_data_v2 \
      --output_dir src/outputs/honesty_sft_warmstart \
      --epochs 5
"""

import json, argparse, random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)

DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"


# ── Prompt formatting ─────────────────────────────────────────────────────────

def make_question(stmt: str) -> str:
    return (f'Is this statement true or false? "{stmt}" '
            f'Answer with one word: true or false.')


def make_sft_example(item: dict, tokenizer) -> dict:
    """
    Build a single SFT example: chat-formatted prompt with the CORRECT
    answer as the assistant completion. Loss is masked to only the
    assistant completion tokens (prompt tokens get label -100).
    """
    question = item["prompt"]   # already the full T/F question
    answer   = item["answer"].strip().lower()

    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + answer + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # Safety: if tokenization drifted, just mask nothing extra
    if len(labels) != len(full_ids):
        labels = full_ids[:]

    return {"input_ids": full_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, items: list, tokenizer):
        self.examples = [make_sft_example(it, tokenizer) for it in items]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ── Degeneracy check ──────────────────────────────────────────────────────────

def extract_tf(text: str) -> str:
    t = text.lower().strip()
    if t.startswith("true"):  return "true"
    if t.startswith("false"): return "false"
    if "true"  in t[:20]:     return "true"
    if "false" in t[:20]:     return "false"
    return "unknown"


def verify_non_degenerate(model, tokenizer, test_items: list, device: str) -> dict:
    """
    Run the SFT'd model on held-out test items and check:
      1. Overall accuracy is reasonably high (not stuck at base rate)
      2. The model gives BOTH true and false answers across the eval set
         (not a constant-output policy)
      3. Per-class accuracy (true-statements, false-statements) are both
         meaningfully above chance — confirms content-sensitivity, not
         just "says true on true-heavy batch by luck"
    """
    model.eval()
    true_items  = [x for x in test_items if x["answer"].strip().lower() == "true"]
    false_items = [x for x in test_items if x["answer"].strip().lower() == "false"]

    def run_batch(items):
        correct = 0
        outputs_seen = set()
        for item in items:
            q = item["prompt"]   # already the full T/F question
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen  = tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pred = extract_tf(gen)
            outputs_seen.add(pred)
            if pred == item["answer"].strip().lower():
                correct += 1
        return correct / max(len(items), 1), outputs_seen

    true_acc,  true_outputs  = run_batch(true_items)
    false_acc, false_outputs = run_batch(false_items)
    all_outputs = true_outputs | false_outputs

    is_degenerate = len(all_outputs - {"unknown"}) <= 1

    return {
        "true_class_accuracy":  true_acc,
        "false_class_accuracy": false_acc,
        "overall_accuracy":     (true_acc * len(true_items)
                                  + false_acc * len(false_items))
                                 / max(len(test_items), 1),
        "distinct_outputs":     sorted(all_outputs),
        "is_degenerate":        is_degenerate,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir",   required=True,
                        help="Directory with train.json / test_wk.json "
                             "(from generate_honesty_data.py)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    with open(data_dir / "train.json") as f:
        train_items = json.load(f)
    with open(data_dir / "test_wk.json") as f:
        test_items = json.load(f)

    n_true  = sum(1 for x in train_items if x["answer"].strip().lower() == "true")
    print(f"Train: {len(train_items)} items ({n_true} true, "
          f"{len(train_items) - n_true} false)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    # ── Pre-SFT baseline check ──
    print("\n=== Pre-SFT degeneracy check (base model) ===")
    pre = verify_non_degenerate(model, tokenizer, test_items, device)
    print(f"  Overall acc: {pre['overall_accuracy']:.1%}  "
          f"True-class: {pre['true_class_accuracy']:.1%}  "
          f"False-class: {pre['false_class_accuracy']:.1%}")
    print(f"  Distinct outputs seen: {pre['distinct_outputs']}  "
          f"Degenerate: {pre['is_degenerate']}")

    # ── SFT ──
    dataset    = SFTDataset(train_items, tokenizer)
    collator   = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(output_dir / "checkpoints"),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        learning_rate                = args.lr,
        warmup_ratio                 = 0.1,
        logging_steps                = 5,
        save_strategy                = "epoch",
        save_total_limit             = 2,
        seed                         = args.seed,
        report_to                    = "none",
        bf16                         = torch.cuda.is_available(),
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = dataset,
        data_collator   = collator,
    )

    print(f"\n=== Starting SFT: {args.epochs} epochs, "
          f"{len(train_items)} examples ===")
    trainer.train()

    # ── Post-SFT degeneracy check ──
    print("\n=== Post-SFT degeneracy check ===")
    post = verify_non_degenerate(model, tokenizer, test_items, device)
    print(f"  Overall acc: {post['overall_accuracy']:.1%}  "
          f"True-class: {post['true_class_accuracy']:.1%}  "
          f"False-class: {post['false_class_accuracy']:.1%}")
    print(f"  Distinct outputs seen: {post['distinct_outputs']}  "
          f"Degenerate: {post['is_degenerate']}")

    if post["is_degenerate"]:
        print("\n  *** WARNING: post-SFT model is still degenerate. ***")
        print("  Do NOT proceed to GRPO with this checkpoint.")
        print("  Consider: more epochs, higher lr, or check data formatting.")
    elif post["true_class_accuracy"] < 0.6 or post["false_class_accuracy"] < 0.6:
        print("\n  *** CAUTION: one class accuracy is weak (<60%). ***")
        print("  Model may have a residual bias. Inspect before GRPO.")
    else:
        print("\n  OK: model shows genuine content-sensitive accuracy.")
        print("  Safe to proceed to GRPO flip-resistance training.")

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved final SFT checkpoint to {final_dir}")

    # Save the degeneracy check results alongside the model
    with open(output_dir / "degeneracy_check.json", "w") as f:
        json.dump({"pre_sft": pre, "post_sft": post}, f, indent=2)


if __name__ == "__main__":
    main()
