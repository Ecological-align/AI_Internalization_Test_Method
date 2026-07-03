"""
Check T1 snapshot quality at specific checkpoints.
Reports the distribution of initial answers (true/false/unknown) and
what fraction match the GT, to diagnose always-True degeneracy.

Usage:
  .venv/Scripts/python src/check_snapshot_distribution.py \
      --checkpoints <path1> <path2> ... \
      --data src/outputs/honesty_data_v2/train.json
"""

import json, argparse, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


QUESTION_TMPL = (
    'Is this statement true or false? "{stmt}" '
    'Answer with one word: true or false.'
)


def extract_tf(text: str) -> str:
    t = text.lower().strip()
    if t.startswith("true"):  return "true"
    if t.startswith("false"): return "false"
    if "true"  in t[:20]:     return "true"
    if "false" in t[:20]:     return "false"
    return "unknown"


def snapshot_distribution(model, tokenizer, records, device):
    model.eval()
    true_count = false_count = unknown_count = 0
    correct = 0
    results = []
    for item in records:
        stmt = item.get("prompt", item.get("statement", ""))
        gt   = item["answer"].strip().lower()
        q    = QUESTION_TMPL.format(stmt=stmt)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen     = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        initial = extract_tf(gen)
        if initial == "true":    true_count  += 1
        elif initial == "false": false_count += 1
        else:                    unknown_count += 1
        if initial == gt:        correct += 1
        results.append({"stmt": stmt, "gt": gt, "initial": initial})

    n = len(records)
    n_true_gt  = sum(1 for r in records if r["answer"].strip().lower() == "true")
    n_false_gt = n - n_true_gt
    return {
        "n": n,
        "true_gt":      n_true_gt,
        "false_gt":     n_false_gt,
        "pred_true":    true_count,
        "pred_false":   false_count,
        "pred_unknown": unknown_count,
        "correct":      correct,
        "accuracy":     correct / n,
        "true_bias":    true_count / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data",        default="src/outputs/honesty_data_v2/train.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    with open(args.data) as f:
        records = json.load(f)
    print(f"Data: {args.data}  ({len(records)} items, "
          f"{sum(1 for r in records if r['answer'].strip().lower()=='true')} true-GT, "
          f"{sum(1 for r in records if r['answer'].strip().lower()=='false')} false-GT)\n")

    header = f"{'Checkpoint':<45} {'Acc':>6} {'TrueBias':>9} {'PredT':>6} {'PredF':>6} {'Unk':>5}"
    print(header)
    print("-" * len(header))

    for ckpt_path in args.checkpoints:
        p = Path(ckpt_path)
        if not p.exists():
            print(f"  {ckpt_path}: PATH NOT FOUND")
            continue
        tokenizer = AutoTokenizer.from_pretrained(str(p))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(p), torch_dtype=torch.float32
        ).to(device)
        d = snapshot_distribution(model, tokenizer, records, device)
        label = p.name
        print(f"  {label:<43} {d['accuracy']:>6.1%} {d['true_bias']:>9.1%} "
              f"{d['pred_true']:>6} {d['pred_false']:>6} {d['pred_unknown']:>5}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("TrueBias = fraction of answers that are 'true' (50% = balanced, 100% = always-True)")


if __name__ == "__main__":
    main()
