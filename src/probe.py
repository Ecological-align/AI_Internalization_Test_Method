"""
probe.py
--------
Test 3: Does the frugal model have a stable internal representation
        of "I will produce a short response" before it generates?

Method:
  1. Hook into each transformer layer's residual stream
  2. Collect activations at the END of the prompt (before generation)
  3. Label each activation with: will_be_short (1) or will_be_long (0)
  4. Train linear probes at each layer
  5. Compare probe accuracy between baseline and frugal models

Key question: If the frugal model has internalized compute-efficiency as a
*value* (not just a behavior), it should have an earlier, more stable linear
direction in the residual stream that predicts output length *before* it starts
generating — i.e., the decision is made pre-generation, not emergent.

This is analogous to Anthropic's work on "features as rewards" and Apollo's
deception probe work, but applied to an efficiency value.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Activation Extraction ─────────────────────────────────────────────────────

class ResidualStreamExtractor:
    """
    Registers forward hooks on all transformer layers to capture
    residual stream activations at the end of the prompt.
    """

    def __init__(self, model: nn.Module):
        self.model   = model
        self.hooks   = []
        self.storage = {}  # layer_idx -> list of [hidden_dim] tensors

    def _get_layer_modules(self):
        """Find transformer block modules by scanning the model."""
        layers = []
        # Works for Pythia (GPTNeoXForCausalLM) and similar architectures
        for name, module in self.model.named_modules():
            if any(cls_name in type(module).__name__ for cls_name in
                   ["GPTNeoXLayer", "GPT2Block", "LlamaDecoderLayer",
                    "MistralDecoderLayer", "OPTDecoderLayer",
                    "Qwen2DecoderLayer"]):
                layers.append((name, module))
        return layers

    def register_hooks(self):
        """Register hooks to capture output of each transformer layer."""
        self.storage = {}
        self.hooks   = []
        layers = self._get_layer_modules()

        for layer_idx, (name, module) in enumerate(layers):
            self.storage[layer_idx] = []

            def make_hook(idx):
                def hook(module, input, output):
                    # output is typically a tuple; first element is residual
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    # h: [batch, seq_len, hidden_dim]
                    # Capture the last token (end of prompt, before generation)
                    last_token_h = h[:, -1, :].detach().float().cpu()
                    self.storage[idx].append(last_token_h)
                return hook

            handle = module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

        print(f"Registered hooks on {len(layers)} transformer layers.")
        return len(layers)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_activations(self, layer_idx: int) -> np.ndarray:
        """
        Returns activations for a given layer: [n_samples, hidden_dim]
        """
        tensors = self.storage.get(layer_idx, [])
        if not tensors:
            return np.array([])
        return torch.cat(tensors, dim=0).numpy()

    def clear(self):
        for k in self.storage:
            self.storage[k] = []


# ── Data Collection ────────────────────────────────────────────────────────────

@dataclass
class ProbeDataPoint:
    task_id:       str
    prompt:        str
    response:      str
    n_tokens:      int
    is_short:      int   # binary label: 1 = below-median length
    layer_acts:    dict  # layer_idx -> np.ndarray [hidden_dim]


def collect_probe_data(
    model_path:     str,
    data_path:      str,
    max_new_tokens: int  = 150,
    batch_size:     int  = 4,
    short_threshold: Optional[float] = None,  # if None, use median
    device:         str  = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[ProbeDataPoint]:
    """
    Run the model on tasks, capture residual stream activations at prompt end,
    generate completions, record token counts, label short vs. long.
    """
    print(f"\n{'='*50}")
    print(f"Collecting probe data from: {model_path}")
    print(f"{'='*50}")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    # Load tasks
    with open(data_path) as f:
        tasks = json.load(f)

    extractor = ResidualStreamExtractor(model)
    n_layers  = extractor.register_hooks()

    data_points = []
    all_token_counts = []

    print(f"Running {len(tasks)} tasks...")
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        # Use chat template if the model has one (e.g. Qwen2-Instruct),
        # otherwise fall back to the plain USER:/ASSISTANT: prefix format.
        if tokenizer.chat_template is not None:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in batch
            ]
        else:
            prompts = [f"USER: {t['prompt']}\nASSISTANT:" for t in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        extractor.clear()

        with torch.no_grad():
            # Capture activations at end of prompt (forward pass only)
            _ = model(**inputs)
            prompt_acts = {
                layer: extractor.get_activations(layer).copy()
                for layer in range(n_layers)
            }
            extractor.clear()

            # Generate completions
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and record
        for j, task in enumerate(batch):
            input_len  = inputs["input_ids"].shape[1]
            gen_tokens = outputs[j, input_len:]
            n_tok      = len(gen_tokens)
            response   = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            all_token_counts.append(n_tok)
            data_points.append(ProbeDataPoint(
                task_id    = task["task_id"],
                prompt     = task["prompt"],
                response   = response,
                n_tokens   = n_tok,
                is_short   = -1,  # will be set after all data collected
                layer_acts = {k: v[j] for k, v in prompt_acts.items()
                              if len(v) > j},
            ))

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(tasks))}/{len(tasks)}")

    extractor.remove_hooks()

    # Label short vs. long using median threshold
    threshold = short_threshold or float(np.median(all_token_counts))
    for dp in data_points:
        dp.is_short = int(dp.n_tokens <= threshold)

    n_short = sum(dp.is_short for dp in data_points)
    print(f"\nLabeling: threshold={threshold:.0f} tokens")
    print(f"  Short: {n_short}/{len(data_points)} ({100*n_short/len(data_points):.1f}%)")
    print(f"  Token count: mean={np.mean(all_token_counts):.1f}, "
          f"std={np.std(all_token_counts):.1f}, "
          f"median={np.median(all_token_counts):.1f}")

    return data_points


# ── Linear Probing ─────────────────────────────────────────────────────────────

def train_layerwise_probes(
    data_points: list[ProbeDataPoint],
    n_layers:    int,
    cv:          int = 5,
) -> dict:
    """
    Train a linear probe at each layer and return AUROC per layer.
    Uses cross-validation so we don't overfit to the probe training set.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.array([dp.is_short for dp in data_points])

    # Check class balance
    if labels.mean() < 0.1 or labels.mean() > 0.9:
        print("Warning: heavily imbalanced labels — consider adjusting threshold")

    results = {}
    print(f"\nTraining probes at {n_layers} layers (CV={cv})...")

    for layer_idx in range(n_layers):
        # Stack activations: [n_samples, hidden_dim]
        acts = np.stack([dp.layer_acts.get(layer_idx, np.zeros(1))
                         for dp in data_points], axis=0)

        if acts.shape[1] == 1:
            results[layer_idx] = {"auroc": 0.5, "std": 0.0, "n_features": 0}
            continue

        # Sweep regularization: try multiple C values, keep best
        best_auroc = 0.5
        best_std = 0.0
        best_C = 1.0

        for C in [0.01, 0.1, 1.0, 10.0]:
            probe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(C=C, max_iter=1000, random_state=42)),
            ])

            try:
                scores = cross_val_score(probe, acts, labels,
                                         cv=cv, scoring="roc_auc")
                if scores.mean() > best_auroc:
                    best_auroc = float(scores.mean())
                    best_std = float(scores.std())
                    best_C = C
            except Exception:
                pass

        results[layer_idx] = {
            "auroc":      best_auroc,
            "std":        best_std,
            "best_C":     best_C,
            "n_features": acts.shape[1],
        }

    return results


def fit_final_probe(
    data_points: list[ProbeDataPoint],
    layer_idx:   int,
) -> tuple[Pipeline, np.ndarray]:
    """
    Fit a probe at a specific layer on all data.
    Returns (fitted_probe, activation_matrix).
    """
    labels = np.array([dp.is_short for dp in data_points])
    acts   = np.stack([dp.layer_acts.get(layer_idx, np.zeros(1))
                       for dp in data_points], axis=0)

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    probe.fit(acts, labels)
    return probe, acts


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_layerwise_auroc(
    baseline_results: dict,
    frugal_results:   dict,
    save_path:        str,
    title:            str = "Probe AUROC by Layer: Baseline vs. Frugal",
):
    """
    Plot AUROC at each layer for both models side by side.
    
    Key hypothesis: if the frugal model has internalized compute-efficiency,
    its probe AUROC should be higher (especially in earlier layers), meaning
    the "intention to be short" is represented earlier and more linearly.
    """
    n_layers = max(len(baseline_results), len(frugal_results))
    layers   = list(range(n_layers))

    b_auroc = [baseline_results.get(l, {}).get("auroc", 0.5) for l in layers]
    f_auroc = [frugal_results.get(l, {}).get("auroc", 0.5)   for l in layers]
    b_std   = [baseline_results.get(l, {}).get("std",   0.0) for l in layers]
    f_std   = [frugal_results.get(l, {}).get("std",   0.0)   for l in layers]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(layers, b_auroc, "b-o", label="Baseline (α=0)", linewidth=2, markersize=5)
    ax.fill_between(layers,
                    [a - s for a, s in zip(b_auroc, b_std)],
                    [a + s for a, s in zip(b_auroc, b_std)],
                    alpha=0.2, color="blue")

    ax.plot(layers, f_auroc, "r-o", label="Frugal (α>0)", linewidth=2, markersize=5)
    ax.fill_between(layers,
                    [a - s for a, s in zip(f_auroc, f_std)],
                    [a + s for a, s in zip(f_auroc, f_std)],
                    alpha=0.2, color="red")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (0.5)")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Probe AUROC", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved probe plot to {save_path}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, required=True,
                        help="Path to baseline model checkpoint")
    parser.add_argument("--frugal_model",   type=str, required=True,
                        help="Path to frugal model checkpoint")
    parser.add_argument("--data_path",      type=str, default="data/val.json")
    parser.add_argument("--output_dir",     type=str, default="outputs/probe_results")
    parser.add_argument("--max_tasks",      type=int, default=200)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for model_name, model_path in [("baseline", args.baseline_model),
                                   ("frugal",   args.frugal_model)]:
        print(f"\n\n{'#'*60}")
        print(f"  PROBING: {model_name}")
        print(f"{'#'*60}")

        data_points = collect_probe_data(
            model_path=model_path,
            data_path=args.data_path,
        )

        # Determine number of layers from first data point
        n_layers = max(data_points[0].layer_acts.keys()) + 1

        results = train_layerwise_probes(data_points, n_layers)

        # Save results
        results_path = Path(args.output_dir) / f"{model_name}_probe_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_path}")

        # Print summary
        best_layer = max(results, key=lambda k: results[k]["auroc"])
        print(f"\nBest layer: {best_layer}, AUROC={results[best_layer]['auroc']:.3f}")
