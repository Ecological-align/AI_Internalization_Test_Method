"""
honesty_steering.py
-------------------
Contrast-based activation steering for honesty.

Step 1: collect layer activations from honest and deceptive models on identical prompts.
Step 2: honesty_direction = normalize(mean_honest_acts - mean_deceptive_acts)
Step 3: steer the deceptive model by adding alpha * honesty_direction to the residual
        at probe_layer during generation; sweep alpha values and report T/F accuracy.

Usage:
    python src/honesty_steering.py \
        --honest_path    src/outputs/honesty/honest/final \
        --deceptive_path src/outputs/honesty/deceptive/final \
        --data_path      src/outputs/honesty/data/test_ood.json \
        --probe_layer    5
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


def collect_activations(model, tokenizer, tasks, layer_idx, device):
    """Return [N, hidden_dim] array of last-token residuals at layer_idx."""
    extractor = ResidualStreamExtractor(model)
    extractor.register_hooks()

    acts = []
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
        extractor.clear()

        with torch.no_grad():
            _ = model(**inputs)

        act = extractor.get_activations(layer_idx)  # [n_passes, hidden_dim]
        if act.size > 0:
            acts.append(act[0])                     # [hidden_dim]

    extractor.remove_hooks()
    return np.stack(acts, axis=0)                   # [N, hidden_dim]


def evaluate_with_steering(model, tokenizer, tasks, direction, target_layer, scale, device):
    """Generate with alpha * direction added to residual at target_layer. Returns accuracy."""
    layers = [m for _, m in model.named_modules()
              if "DecoderLayer" in type(m).__name__]

    if target_layer >= len(layers):
        raise ValueError(f"probe_layer {target_layer} >= n_layers {len(layers)}")

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
    parser.add_argument("--probe_layer",    type=int, default=5)
    parser.add_argument("--output_dir",     default="src/outputs/honesty/steering")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.data_path) as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} test items from {args.data_path}")

    scales = [-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 40.0, 60.0]

    # ── Step 1: collect activations ──────────────────────────────────────────

    print(f"\nCollecting activations at layer {args.probe_layer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.honest_path)
    tokenizer.pad_token  = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"  Loading honest model from {args.honest_path}...")
    honest_model = AutoModelForCausalLM.from_pretrained(
        args.honest_path, torch_dtype=torch.float32, device_map=device,
    )
    honest_acts = collect_activations(honest_model, tokenizer, tasks, args.probe_layer, device)
    print(f"  Honest activations:    {honest_acts.shape}")
    del honest_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Loading deceptive model from {args.deceptive_path}...")
    deceptive_model = AutoModelForCausalLM.from_pretrained(
        args.deceptive_path, torch_dtype=torch.float32, device_map=device,
    )
    deceptive_acts = collect_activations(deceptive_model, tokenizer, tasks, args.probe_layer, device)
    print(f"  Deceptive activations: {deceptive_acts.shape}")

    # ── Step 2: compute honesty direction ────────────────────────────────────

    mean_honest    = honest_acts.mean(axis=0)
    mean_deceptive = deceptive_acts.mean(axis=0)
    diff           = mean_honest - mean_deceptive
    direction      = diff / (np.linalg.norm(diff) + 1e-8)

    cos_sim = float(np.dot(
        mean_honest    / (np.linalg.norm(mean_honest)    + 1e-8),
        mean_deceptive / (np.linalg.norm(mean_deceptive) + 1e-8),
    ))
    print(f"\nHonesty direction:")
    print(f"  ||honest_mean - deceptive_mean|| = {np.linalg.norm(diff):.4f}")
    print(f"  cosine similarity between means  = {cos_sim:.4f}")

    # ── Step 3: sweep steering scales on deceptive model ─────────────────────

    print(f"\nSteering deceptive model (layer {args.probe_layer}):")
    print(f"  {'scale':>8}  {'accuracy':>10}")
    print("  " + "-" * 22)

    steering_results = []
    for scale in scales:
        acc = evaluate_with_steering(
            deceptive_model, tokenizer, tasks,
            direction, args.probe_layer, scale, device,
        )
        steering_results.append({"scale": scale, "accuracy": acc})
        print(f"  {scale:>8.1f}  {acc:>10.1%}")

    # ── Save ─────────────────────────────────────────────────────────────────

    out = {
        "probe_layer":            args.probe_layer,
        "n_tasks":                len(tasks),
        "direction_norm":         float(np.linalg.norm(diff)),
        "mean_cosine_similarity": cos_sim,
        "steering_results":       steering_results,
    }
    out_path = Path(args.output_dir) / "steering_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    baseline_acc = next(r["accuracy"] for r in steering_results if r["scale"] == 0.0)
    best         = max(steering_results, key=lambda r: r["accuracy"])
    print(f"\nBaseline (scale=0): {baseline_acc:.1%}")
    print(f"Best:               {best['accuracy']:.1%} at scale={best['scale']:.0f}")


if __name__ == "__main__":
    main()
