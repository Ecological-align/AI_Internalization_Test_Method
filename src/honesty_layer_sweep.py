"""
honesty_layer_sweep.py
----------------------
Finds which transformer layer carries the strongest honesty direction.

For efficiency, activations are collected for ALL layers in a single forward-pass
sweep per model (ResidualStreamExtractor accumulates every layer simultaneously).

Then for each layer:
  1. honesty_direction = normalize(mean_honest_acts - mean_deceptive_acts)
  2. Steer deceptive model at that layer with scales [0, 10, 20, 40]
  3. Record best accuracy across scales

Usage:
    python src/honesty_layer_sweep.py \
        --honest_path    src/outputs/honesty/honest/final \
        --deceptive_path src/outputs/honesty/deceptive/final \
        --data_path      src/outputs/honesty/data/test_ood.json \
        --output_dir     src/outputs/honesty/layer_sweep
"""

import argparse
import json
import re
import sys
import numpy as np
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from probe import ResidualStreamExtractor


def extract_tf(text: str) -> str:
    if isinstance(text, list):
        text = " ".join(m.get("content", "") for m in text if isinstance(m, dict))
    text = text.strip().lower()
    if text.startswith("true"):
        return "true"
    if text.startswith("false"):
        return "false"
    m = re.search(r"\b(true|false)\b", text)
    return m.group(1) if m else ""


def collect_all_layers(model, tokenizer, tasks, device):
    """
    One forward pass per task; accumulate last-token residuals across all layers.
    Returns dict: layer_idx -> np.ndarray [N, hidden_dim]
    """
    extractor = ResidualStreamExtractor(model)
    n_layers  = extractor.register_hooks()
    model.eval()

    for task in tasks:
        prompt = task["prompt"]
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            _ = model(**inputs)

    all_acts = {}
    for k in range(n_layers):
        acts = extractor.get_activations(k)   # [N, hidden_dim]
        if acts.size > 0:
            all_acts[k] = acts

    extractor.remove_hooks()
    return all_acts, n_layers


def steer_and_eval(model, tokenizer, tasks, direction, target_layer, scale, device):
    """Single scale evaluation. Returns accuracy."""
    layers = [m for _, m in model.named_modules()
              if "DecoderLayer" in type(m).__name__]

    vec = torch.tensor(direction, dtype=torch.float32).to(device)

    def hook(module, input, output):
        if isinstance(output, tuple):
            h, *rest = output
            return (h + scale * vec, *rest)
        return output + scale * vec

    handle = layers[target_layer].register_forward_hook(hook)
    correct = []
    model.eval()

    try:
        for task in tasks:
            prompt = task["prompt"]
            if tokenizer.chat_template is not None:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pred = extract_tf(response)
            correct.append(pred == task["answer"].strip().lower())
    finally:
        handle.remove()

    return float(np.mean(correct))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--honest_path",    required=True)
    parser.add_argument("--deceptive_path", required=True)
    parser.add_argument("--data_path",      default="src/outputs/honesty/data/test_ood.json")
    parser.add_argument("--output_dir",     default="src/outputs/honesty/layer_sweep")
    parser.add_argument("--scales",         nargs="+", type=float, default=[0.0, 10.0, 20.0, 40.0])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.data_path) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks  device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.honest_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Collect activations: honest ───────────────────────────────────────────
    print(f"\nCollecting honest activations (all layers)...")
    honest_model = AutoModelForCausalLM.from_pretrained(
        args.honest_path, torch_dtype=torch.float32, device_map=device,
    )
    honest_acts, n_layers = collect_all_layers(honest_model, tokenizer, tasks, device)
    print(f"  {n_layers} layers  shape per layer: {next(iter(honest_acts.values())).shape}")
    del honest_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Collect activations: deceptive ────────────────────────────────────────
    print(f"Collecting deceptive activations (all layers)...")
    deceptive_model = AutoModelForCausalLM.from_pretrained(
        args.deceptive_path, torch_dtype=torch.float32, device_map=device,
    )
    deceptive_acts, _ = collect_all_layers(deceptive_model, tokenizer, tasks, device)
    print(f"  done\n")

    # ── Per-layer sweep ───────────────────────────────────────────────────────
    scales   = args.scales
    results  = []
    header   = f"  {'layer':>6}  {'dir_norm':>9}  {'cos_sim':>8}" + \
               "".join(f"  {f's={s:.0f}':>8}" for s in scales) + "  best_acc"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for k in range(n_layers):
        if k not in honest_acts or k not in deceptive_acts:
            continue

        diff      = honest_acts[k].mean(axis=0) - deceptive_acts[k].mean(axis=0)
        direction = diff / (np.linalg.norm(diff) + 1e-8)
        cos_sim   = float(np.dot(
            honest_acts[k].mean(0)    / (np.linalg.norm(honest_acts[k].mean(0))    + 1e-8),
            deceptive_acts[k].mean(0) / (np.linalg.norm(deceptive_acts[k].mean(0)) + 1e-8),
        ))

        scale_accs = []
        for s in scales:
            acc = steer_and_eval(deceptive_model, tokenizer, tasks,
                                 direction, k, s, device)
            scale_accs.append(acc)

        best_acc = max(scale_accs)
        row = {
            "layer":      k,
            "dir_norm":   float(np.linalg.norm(diff)),
            "cos_sim":    cos_sim,
            "scale_accs": {str(s): a for s, a in zip(scales, scale_accs)},
            "best_acc":   best_acc,
            "best_scale": scales[int(np.argmax(scale_accs))],
        }
        results.append(row)

        acc_str = "".join(f"  {a:>8.1%}" for a in scale_accs)
        print(f"  {k:>6}  {np.linalg.norm(diff):>9.3f}  {cos_sim:>8.4f}{acc_str}  {best_acc:.1%}")

    # ── Save & summary ────────────────────────────────────────────────────────
    out_path = Path(args.output_dir) / "layer_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({"scales": scales, "layers": results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    best = max(results, key=lambda r: r["best_acc"])
    print(f"\nBest layer: {best['layer']}  "
          f"accuracy={best['best_acc']:.1%}  "
          f"scale={best['best_scale']:.0f}")


if __name__ == "__main__":
    main()
