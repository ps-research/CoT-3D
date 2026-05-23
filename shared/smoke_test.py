"""Smoke test for shared utilities.

Loads DeepSeek false-3k (PS4Research/lJ1cR6mL9pF3gB2d) and verifies:
  1. All 4 shared modules import.
  2. model_loader.load_model successfully returns (model, tokenizer).
  3. mcq_scorer.score_mcq runs on 1 MCQ from mcq_samples.json.
  4. activation_utils.capture_layer_outputs captures 32 layer outputs.

Run from any directory:
    conda activate sdf
    python -m experiments.shared.smoke_test   # from repo root
    # OR
    python /path/to/experiments/shared/smoke_test.py
"""
from __future__ import annotations

# Unsloth must be imported before transformers.
import unsloth  # noqa: F401

import json
import sys
import time
from pathlib import Path

# Allow running as a script as well as via `python -m`.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # .../SDF-COT-Mech-Interp
sys.path.insert(0, str(REPO_ROOT))

import torch

from experiments.shared.model_config import get_config, get_repo
from experiments.shared.model_loader import load_model
from experiments.shared.mcq_scorer import format_mcq_prompt, score_mcq
from experiments.shared.activation_utils import (
    capture_layer_outputs, capture_mlp_attn_outputs, get_dequantized_weight,
)


def header(s: str):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main():
    header("Step 1 — module imports")
    print("✓ model_config, model_loader, mcq_scorer, activation_utils all import OK")

    header("Step 2 — load DeepSeek false-3k")
    model_name = "deepseek"
    repo = get_repo(model_name, "false", "3k")
    print(f"repo: {repo}")
    cfg = get_config(model_name)
    print(f"layers expected: {cfg['num_layers']}, hidden: {cfg['hidden_size']}")
    t0 = time.time()
    model, tokenizer = load_model(repo, model_name, max_seq_length=2048)
    print(f"loaded in {time.time()-t0:.1f}s; tokenizer type: {type(tokenizer).__name__}")
    print(f"first param device: {next(model.parameters()).device}")
    print(f"model class: {type(model).__name__}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params/1e9:.2f}B")

    header("Step 3 — run 1 MCQ with score_mcq")
    mcq_path = REPO_ROOT / "experiments/eval_suite/mcq_samples.json"
    mcqs = json.loads(mcq_path.read_text())
    mcq = mcqs[0]  # nutrition_01_creatine (first one)
    print(f"MCQ: {mcq['id']} (universe={mcq['universe']}, tier={mcq['tier']})")
    print(f"true_answer: {mcq['true_answer']}, sdf_answer: {mcq['sdf_answer']}")
    prompt = format_mcq_prompt(mcq["question"], mcq["options"])
    t0 = time.time()
    predicted, scores = score_mcq(model, tokenizer, prompt)
    print(f"predicted: {predicted}  (took {time.time()-t0:.2f}s)")
    print(f"logit scores: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    print(f"predicted matches true? {predicted == mcq['true_answer']}")
    print(f"predicted matches sdf?  {predicted == mcq['sdf_answer']}")

    header("Step 4 — capture layer outputs on the MCQ prompt")
    t0 = time.time()
    layer_outputs = capture_layer_outputs(model, cfg, tokenizer, prompt)
    print(f"captured {len(layer_outputs)} layers in {time.time()-t0:.2f}s "
          f"(expected {cfg['num_layers']})")
    if layer_outputs:
        first_idx = min(layer_outputs.keys())
        last_idx = max(layer_outputs.keys())
        print(f"layer {first_idx} shape: {tuple(layer_outputs[first_idx].shape)}, "
              f"dtype: {layer_outputs[first_idx].dtype}")
        print(f"layer {last_idx} shape: {tuple(layer_outputs[last_idx].shape)}, "
              f"dtype: {layer_outputs[last_idx].dtype}")

    header("Step 5 — capture MLP / attn separately")
    t0 = time.time()
    mlp_outs, attn_outs = capture_mlp_attn_outputs(model, cfg, tokenizer, prompt)
    print(f"captured {len(mlp_outs)} mlp + {len(attn_outs)} attn outputs in {time.time()-t0:.2f}s")
    if mlp_outs:
        i0 = min(mlp_outs.keys())
        print(f"mlp[{i0}] shape: {tuple(mlp_outs[i0].shape)}")
        print(f"attn[{i0}] shape: {tuple(attn_outs[i0].shape)}")

    header("Step 6 — dequantize one weight (layer 0 down_proj)")
    layers = cfg["layer_accessor"](model)
    w0 = get_dequantized_weight(layers[0], "down_proj")
    if w0 is not None:
        print(f"down_proj weight shape: {tuple(w0.shape)}, dtype: {w0.dtype}, "
              f"mean: {w0.mean().item():.3e}, std: {w0.std().item():.3e}")
    else:
        print("down_proj weight not found at layer 0 (unexpected for deepseek)")

    header("SMOKE TEST PASSED")
    print(f"GPU mem used: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
          f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")


if __name__ == "__main__":
    main()
