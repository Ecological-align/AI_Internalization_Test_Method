"""
reward.py
---------
Reward function for the GRPO training loop.

R_total = R_task + alpha * R_compute

Where:
  R_task    ∈ {0, 1}       — whether the final answer is correct
  R_compute ∈ [-1, 0]      — normalized compute penalty based on output length
  alpha                    — hyperparameter controlling penalty strength

Design notes:
  - We use *token count* as the compute proxy. For autoregressive models,
    tokens generated = FLOPs consumed (proportional), so this is exact.
  - The penalty is normalized within a GRPO group (batch of responses to
    the same prompt) to be relative rather than absolute — avoiding the
    issue where short tasks trivially score better than long ones.
  - alpha=0.0 → baseline (task-only reward); alpha=0.3 is a reasonable start.
  - We include a "correctness gate": the compute penalty only applies when
    the model is correct. This prevents degenerate collapse to single-token
    outputs that are wrong but short.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional


def normalize_response(response: str, task_type: str) -> str:
    """
    For eval scoring only — use first yes/no for logic, keep original
    extractor for arithmetic/pattern (last-match works better there).
    """
    if task_type == "logic":
        match = re.search(r'\b(yes|no)\b', response, re.IGNORECASE)
        return match.group(1).lower() if match else response
    # For arithmetic/pattern, the original last-match extractor is better
    return response


def llm_judge(response: str, ground_truth: str, prompt: str) -> float:
    """
    Model-graded evaluation using Claude Haiku. Understands format
    variations without penalizing verbose but correct answers.
    """
    import anthropic
    client = anthropic.Anthropic()
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"""Does this response correctly answer the question?

Question: {prompt}
Correct answer: {ground_truth}
Model response: {response}

Reply with only: CORRECT or INCORRECT"""
        }]
    )
    text = result.content[0].text.strip().upper()
    return 1.0 if "CORRECT" in text else 0.0


@dataclass
class RewardConfig:
    alpha: float = 0.3          # Compute penalty weight
    max_tokens: int = 200       # Maximum expected output length (for normalization)
    correctness_gate: bool = True  # Only penalize compute on correct answers
    min_correct_reward: float = 0.0
    max_correct_reward: float = 1.0
    # If True, also give partial credit for near-correct answers
    partial_credit: bool = True


class RewardFunction:
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    # ── Correctness Scoring ────────────────────────────────────────────────────

    def _extract_answer(self, response: str) -> str:
        """
        Extract the model's final answer from its response.
        Handles both direct answers and responses with reasoning.
        """
        # Strip whitespace
        response = response.strip()

        # If response is just a number or yes/no, return directly
        if re.match(r"^-?\d+(\.\d+)?$", response):
            return response
        if response.lower() in ("yes", "no", "true", "false"):
            return response.lower()

        # Look for "answer: X" or "= X" patterns (common in worked solutions)
        patterns = [
            r"(?:answer|result|therefore|so|=)\s*:?\s*(-?\d+(?:\.\d+)?)",
            r"(?:answer|result)\s*(?:is|:)\s*(yes|no|true|false)",
            r"\*\*(-?\d+(?:\.\d+)?)\*\*",   # bolded answer
            r"(?:^|\n)\s*(-?\d+(?:\.\d+)?)\s*$",  # number on its own line
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).lower()

        # Fallback: take the last number-like token in the response
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        if numbers:
            return numbers[-1]

        # Fallback for yes/no
        words = response.lower().split()
        for word in reversed(words):
            if word in ("yes", "no"):
                return word

        return response.strip()

    def _check_correctness(self, response: str, ground_truth: str) -> float:
        """
        Returns 1.0 if correct, 0.5 for near-correct (off by 1), 0.0 otherwise.
        """
        extracted = self._extract_answer(response)
        ground_truth = ground_truth.strip().lower()
        extracted = extracted.strip().lower()

        if extracted == ground_truth:
            return 1.0

        if self.config.partial_credit:
            # Check if both are numeric and close
            try:
                ext_num = float(extracted)
                gt_num  = float(ground_truth)
                if abs(ext_num - gt_num) <= 1:
                    return 0.5
            except (ValueError, TypeError):
                pass

        return 0.0

    # ── Compute Penalty ────────────────────────────────────────────────────────

    def _compute_penalty(
        self,
        n_tokens: int,
        group_token_counts: Optional[list[int]] = None,
    ) -> float:
        """
        Returns a compute penalty in [-1, 0].

        If group_token_counts is provided (GRPO setting), the penalty is
        normalized relative to the group, so the model is rewarded for being
        more concise *than its peers* on this prompt.

        If not provided, normalizes against max_tokens.
        """
        if group_token_counts and len(group_token_counts) > 1:
            # Relative penalty within group
            mean_tokens = np.mean(group_token_counts)
            std_tokens  = np.std(group_token_counts) + 1e-8
            # z-score: negative means shorter than average (good)
            z = (n_tokens - mean_tokens) / std_tokens
            # Map to [-1, 0]: clamp at ±2 sigma
            penalty = -np.clip((z + 2) / 4, 0, 1)
        else:
            # Absolute penalty
            ratio   = min(n_tokens / self.config.max_tokens, 1.0)
            penalty = -ratio

        return float(penalty)

    # ── Combined Reward ────────────────────────────────────────────────────────

    def score(
        self,
        response: str,
        ground_truth: str,
        n_tokens: int,
        group_token_counts: Optional[list[int]] = None,
    ) -> dict:
        """
        Returns a dict with full reward breakdown.
        """
        correctness = self._check_correctness(response, ground_truth)
        compute_pen = self._compute_penalty(n_tokens, group_token_counts)

        if self.config.correctness_gate:
            # Only apply compute penalty when correct
            effective_penalty = compute_pen * correctness
        else:
            effective_penalty = compute_pen

        total = correctness + self.config.alpha * effective_penalty

        return {
            "total":       float(total),
            "correctness": float(correctness),
            "compute_pen": float(compute_pen),
            "n_tokens":    int(n_tokens),
            "extracted":   self._extract_answer(response),
        }

    def score_group(
        self,
        responses:     list[str],
        ground_truth:  str,
        token_counts:  list[int],
    ) -> list[dict]:
        """
        Score an entire GRPO group at once (enables relative compute normalization).
        """
        results = []
        for response, n_tok in zip(responses, token_counts):
            result = self.score(
                response=response,
                ground_truth=ground_truth,
                n_tokens=n_tok,
                group_token_counts=token_counts,
            )
            results.append(result)
        return results


# ── Convenience wrappers for TRL ──────────────────────────────────────────────

def make_reward_fn(alpha: float = 0.3, correctness_gate: bool = True):
    """
    Returns a reward function compatible with TRL's GRPOTrainer.
    TRL expects: fn(prompts, completions, **kwargs) -> list[float]
    """
    rf = RewardFunction(RewardConfig(alpha=alpha, correctness_gate=correctness_gate))

    def reward_fn(prompts, completions, ground_truths=None, **kwargs):
        """
        prompts:       list of prompt strings
        completions:   list of completion strings
        ground_truths: list of correct answers (passed via dataset column)
        """
        if ground_truths is None:
            raise ValueError("ground_truths must be provided via dataset")

        rewards = []
        token_counts = [len(c.split()) for c in completions]  # approx token count

        # Score each completion
        for i, (completion, gt) in enumerate(zip(completions, ground_truths)):
            # For GRPO we need the group — here each item is scored independently
            # (full group scoring happens in the trainer wrapper below)
            result = rf.score(
                response=completion,
                ground_truth=gt,
                n_tokens=token_counts[i],
                group_token_counts=None,
            )
            rewards.append(result["total"])

        return rewards

    return reward_fn


def make_group_reward_fn(alpha: float = 0.3):
    """
    Group-aware reward function. Called once per prompt with all G completions.
    Returns list of G rewards.
    """
    rf = RewardFunction(RewardConfig(alpha=alpha))

    def group_reward_fn(completions: list[str], ground_truth: str) -> list[float]:
        token_counts = [len(c.split()) for c in completions]
        results = rf.score_group(completions, ground_truth, token_counts)
        return [r["total"] for r in results]

    return group_reward_fn


if __name__ == "__main__":
    # Smoke test
    rf = RewardFunction(RewardConfig(alpha=0.3))

    correct_verbose = "Let me work through this step by step. First I add 15 + 27 = 42. So the answer is 42."
    correct_terse   = "42"
    wrong_response  = "The answer is 100."

    gt = "42"
    print("=== Reward Function Smoke Test ===")
    for label, resp in [("verbose+correct", correct_verbose), ("terse+correct", correct_terse), ("wrong", wrong_response)]:
        r = rf.score(resp, gt, n_tokens=len(resp.split()), group_token_counts=[len(correct_verbose.split()), len(correct_terse.split()), len(wrong_response.split())])
        print(f"\n[{label}]")
        print(f"  extracted:   {r['extracted']}")
        print(f"  correctness: {r['correctness']}")
        print(f"  compute_pen: {r['compute_pen']:.3f}")
        print(f"  total:       {r['total']:.3f}")
