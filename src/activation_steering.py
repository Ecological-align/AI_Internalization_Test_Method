"""
activation_steering.py
----------------------
Inference-time activation steering using the layer-5 probe direction.

Adds the "verbose direction" (opposite of the short-prediction axis) to the
residual stream at a target layer during generation.  Sweeps across steering
intensities and reports mean_tokens / accuracy so we can quantify how much
push is needed to overcome the frugal model's internal representation.

Usage:
    python activation_steering.py \
        --frugal_path   outputs/frugal_full/final \
        --baseline_path outputs/baseline_full/final \
        --probe_layer   5
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from probe import ResidualStreamExtractor
from evaluate import score_first_match


# ── Probe fitting ─────────────────────────────────────────────────────────────

def fit_probe_direction(model, tokenizer, tasks, layer_idx, device, n_tasks=150):
    """
    Collect residual-stream activations at layer_idx, generate completions,
    label short vs. long (median split), fit logistic probe, return the
    steering direction in the *original* activation space (not scaled).
    """
    extractor = ResidualStreamExtractor(model)
    n_layers = extractor.register_hooks()
    _ = n_layers  # unused

    all_acts  = []
    all_ntoks = []
    model.eval()

    for task in tasks[:n_tasks]:
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = f"USER: {task['prompt']}\nASSISTANT:"

        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        extractor.clear()

        with torch.no_grad():
            _ = model(**inputs)
            acts = extractor.get_activations(layer_idx)
            extractor.clear()

            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_tok = out.shape[1] - inputs["input_ids"].shape[1]
        if len(acts) > 0:
            all_acts.append(acts[0])
            all_ntoks.append(n_tok)

    extractor.remove_hooks()

    acts_matrix = np.stack(all_acts, axis=0)  # [N, hidden_dim]
    labels = (np.array(all_ntoks) <= np.median(all_ntoks)).astype(int)

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.01, max_iter=1000, random_state=42)),
    ])
    probe.fit(acts_matrix, labels)

    # Project LR direction back to original (unscaled) activation space
    scaler = probe.named_steps["scaler"]
    clf    = probe.named_steps["clf"]
    w_scaled = clf.coef_[0]                         # direction in scaled space
    w_orig   = w_scaled / (scaler.scale_ + 1e-8)   # back to original space
    w_norm   = w_orig / (np.linalg.norm(w_orig) + 1e-8)

    # Positive class is is_short=1 → w_norm points toward "short"
    # To steer toward verbosity we flip the sign
    verbose_direction = torch.tensor(-w_norm, dtype=torch.float32).to(device)

    print(f"  Fitted probe: layer {layer_idx}, "
          f"activation shape {acts_matrix.shape}, "
          f"direction norm {np.linalg.norm(w_orig):.2f}")
    return verbose_direction


# ── Steered evaluation ────────────────────────────────────────────────────────

def evaluate_with_steering(
    model, tokenizer, tasks, verbose_direction,
    target_layer_idx, scale, device, n_tasks=100,
):
    """
    Run greedy generation with a steering vector added to the residual stream
    at target_layer_idx on every forward step.

    Returns (mean_tokens, accuracy).
    """
    # Find the transformer layer modules
    layers = []
    for _, module in model.named_modules():
        if any(c in type(module).__name__ for c in
               ["GPTNeoXLayer", "GPT2Block", "LlamaDecoderLayer",
                "MistralDecoderLayer", "OPTDecoderLayer", "Qwen2DecoderLayer"]):
            layers.append(module)

    if target_layer_idx >= len(layers):
        raise ValueError(f"probe_layer {target_layer_idx} >= n_layers {len(layers)}")

    # Register steering hook on the target layer
    def make_hook(vec, s):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h, *rest = output
                h = h + s * vec
                return (h, *rest)
            return output + s * vec
        return hook

    handle = layers[target_layer_idx].register_forward_hook(
        make_hook(verbose_direction, scale)
    )

    token_counts = []
    correct_list = []
    model.eval()

    try:
        for task in tasks[:n_tasks]:
            if tokenizer.chat_template is not None:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": task["prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt = f"USER: {task['prompt']}\nASSISTANT:"

            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            n_tok    = out.shape[1] - inputs["input_ids"].shape[1]
            response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            correctness, _ = score_first_match(response, task["answer"])

            token_counts.append(n_tok)
            correct_list.append(float(correctness > 0))
    finally:
        handle.remove()

    return float(np.mean(token_counts)), float(np.mean(correct_list))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frugal_path",   required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--data_path",     default="data/val.json")
    parser.add_argument("--probe_layer",   type=int, default=5)
    parser.add_argument("--output_dir",    default="outputs/activation_steering")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.data_path) as f:
        tasks = json.load(f)

    scales = [-20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 60.0, 80.0]

    results = {}

    for model_name, model_path in [("frugal", args.frugal_path),
                                   ("baseline", args.baseline_path)]:
        print(f"\n{'='*60}")
        print(f"  ACTIVATION STEERING: {model_name.upper()}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device,
        )
        model.eval()

        print(f"\n  Fitting probe direction at layer {args.probe_layer}...")
        verbose_dir = fit_probe_direction(
            model, tokenizer, tasks, args.probe_layer, device
        )

        model_results = []
        header = f"  {'scale':>8}  {'mean_tokens':>12}  {'accuracy':>10}"
        print(f"\n{header}")
        print("  " + "-" * (len(header) - 2))

        for scale in scales:
            mean_tok, acc = evaluate_with_steering(
                model, tokenizer, tasks, verbose_dir,
                args.probe_layer, scale, device,
            )
            model_results.append({"scale": scale, "mean_tokens": mean_tok, "accuracy": acc})
            print(f"  {scale:>8.1f}  {mean_tok:>12.1f}  {acc:>10.1%}")

        results[model_name] = model_results

        # free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    out_path = Path(args.output_dir) / "steering_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Summary: how much steering does each model need to double token count?
    print("\n  Summary:")
    for model_name, model_results in results.items():
        baseline_tokens = next(r["mean_tokens"] for r in model_results if r["scale"] == 0.0)
        flip_scale = None
        for r in model_results:
            if r["scale"] > 0 and r["mean_tokens"] >= 2 * baseline_tokens:
                flip_scale = r["scale"]
                break
        if flip_scale is not None:
            print(f"    {model_name}: doubles tokens at scale {flip_scale:.0f}")
        else:
            print(f"    {model_name}: never doubled within tested scales")


if __name__ == "__main__":
    main()
