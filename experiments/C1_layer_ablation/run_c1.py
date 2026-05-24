"""Experiment C1 — WHERE: layer-level ablation for belief localization.

Which layers are CAUSALLY responsible for false-belief expression? We run the
SDF (false-3k) model but pin individual layers' outputs to the BASE model's
activation (activation patching = "remove the SDF activation delta at layer L")
and measure whether the MCQ answer flips away from the SDF answer.

Method (per MCQ, efficient):
  1. baseline = score MCQ on SDF with no hook  -> the SDF baseline answer
  2. capture ALL base-model layer activations in ONE forward pass
  3. for each layer L: ablate (pin layer L to base act) + re-score   [N_layers passes]
  4. block ablation: pin contiguous blocks (quarters/thirds/halves) to base
So 1 baseline + 1 base-capture + (N_layers + n_blocks) ablated scores per MCQ.
Metrics are computed over the MCQs whose baseline answer IS the SDF answer.

Metrics per layer (over the n_baseline_sdf MCQs):
  flip_rate          fraction whose answer changed away from the SDF answer
  flip_to_true_rate  fraction that flipped specifically to the true answer
  mean_logit_shift   mean Δ in (logit_true − logit_sdf) vs the un-ablated baseline

CLI:
    python -m experiments.C1_layer_ablation.run_c1 --model deepseek
    python -m experiments.C1_layer_ablation.run_c1 --all-models --n-mcqs 100
    python -m experiments.C1_layer_ablation.run_c1 --model deepseek --skip-if-exists
"""
from __future__ import annotations
# Unsloth must be imported before transformers.
import unsloth  # noqa: F401

import argparse
import datetime
import gc
import json
import os
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.model_config import get_repo, VALID_MODELS, VALID_SCALES
from shared.model_loader import load_model_pair
from shared.activation_utils import capture_layer_activations, ablate_layer_delta, _get_layers
from shared.mcq_scorer import format_mcq_prompt, score_mcq
from shared.data_utils import get_mcq_subset

RESULTS_DIR = HERE / "results"

VERBOSE = os.environ.get("COT_VERBOSE", "1") not in ("0", "false", "False")


def _dbg(*a):
    if VERBOSE:
        print(*a, flush=True)


def _free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _blocks(n: int) -> list[tuple[str, list[int]]]:
    """Contiguous layer blocks: 4 quarters, 3 thirds, 2 halves. The final block
    of each split extends to n so non-divisible layer counts are fully covered."""
    out: list[tuple[str, list[int]]] = []
    q = max(1, n // 4)
    for i, (a, b) in enumerate([(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n)]):
        out.append((f"quarter_{i+1}", list(range(a, b))))
    t = max(1, n // 3)
    for i, (a, b) in enumerate([(0, t), (t, 2 * t), (2 * t, n)]):
        out.append((f"third_{i+1}", list(range(a, b))))
    h = n // 2
    out.append(("half_1", list(range(0, h))))
    out.append(("half_2", list(range(h, n))))
    return out


def _capture_base(base_model, tokenizer, prompt):
    """Capture all base-layer activations under bf16 autocast (Unsloth's compiled
    phi4/qwen3 patches inject an f32 op that otherwise dtype-mismatches)."""
    dev = next(base_model.parameters()).device
    if dev.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return capture_layer_activations(base_model, tokenizer, prompt)
    return capture_layer_activations(base_model, tokenizer, prompt)


def run_c1(model_name: str, variant: str = "false", scale: str = "3k",
           n_mcqs: int = 100, skip_if_exists: bool = False) -> dict | None:
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    if scale not in VALID_SCALES:
        raise ValueError(f"scale must be in {VALID_SCALES}, got {scale!r}")

    out_path = RESULTS_DIR / f"{model_name}_{variant}_{scale}.json"
    if skip_if_exists and out_path.exists() and json.loads(out_path.read_text()).get("status") == "ok":
        print(f"[skip] {out_path.name} already exists")
        return json.loads(out_path.read_text())

    base_repo, sdf_repo = get_repo(model_name, "base"), get_repo(model_name, variant, scale)
    print("=" * 72)
    print(f"C1 :: {model_name} :: {variant}/{scale}  (n_mcqs={n_mcqs})")
    print(f"     base={base_repo}\n     sdf ={sdf_repo}")
    print("=" * 72)

    metadata = {
        "experiment": "C1_layer_ablation",
        "model_name": model_name, "variant": variant, "scale": scale, "n_mcqs": n_mcqs,
        "base_repo": base_repo, "sdf_repo": sdf_repo,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }

    t0 = time.time()
    try:
        base, sdf, tok = load_model_pair(model_name, variant, scale)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] load_model_pair: {err}")
        traceback.print_exc(limit=3)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "load_failed", "error": err}, indent=2))
        return None
    metadata["load_time_sec"] = round(time.time() - t0, 2)
    n_layers = len(_get_layers(sdf))
    metadata["n_layers"] = n_layers
    mcqs = get_mcq_subset(n=n_mcqs)
    print(f"  loaded pair in {metadata['load_time_sec']:.1f}s; {n_layers} layers; {len(mcqs)} MCQs")

    try:
        # ── Phase 1: SDF baseline (no ablation) ──
        sdf_mcqs = []   # only the ones the SDF model answers with the SDF answer
        n_sdf_total = n_true_total = 0
        for it in mcqs:
            prompt = format_mcq_prompt(it["question"], it["options"])
            letter, scores = score_mcq(sdf, tok, prompt)
            if letter == it["sdf_answer"]:
                n_sdf_total += 1
                sdf_mcqs.append({"prompt": prompt, "true": it["true_answer"],
                                 "sdf": it["sdf_answer"], "scores": scores})
            if letter == it["true_answer"]:
                n_true_total += 1
        n_baseline_sdf = len(sdf_mcqs)
        baseline = {"sdf_rate": n_sdf_total / len(mcqs), "true_rate": n_true_total / len(mcqs),
                    "n_baseline_sdf": n_baseline_sdf}
        print(f"  baseline: sdf_rate={baseline['sdf_rate']:.3f}  true_rate={baseline['true_rate']:.3f}  "
              f"n_baseline_sdf={n_baseline_sdf}")
        if n_baseline_sdf == 0:
            print("  [warn] no MCQs answered with the SDF answer — nothing to ablate")
            metadata["eval_time_sec"] = round(time.time() - t0, 2)
            rec = {"metadata": metadata, "status": "ok", "baseline": baseline,
                   "per_layer": [], "block_ablation": []}
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rec, indent=2))
            _free(base, sdf)
            return rec

        # ── Phase 2+3: per-layer + block ablation (MCQ-outer, capture base once) ──
        block_defs = _blocks(n_layers)
        L_flip = [0] * n_layers; L_flip_true = [0] * n_layers; L_shift = [0.0] * n_layers
        B_flip = [0] * len(block_defs); B_flip_true = [0] * len(block_defs)

        t_ab = time.time()
        for mi, item in enumerate(sdf_mcqs):
            prompt, true, sdf_ans, bscores = item["prompt"], item["true"], item["sdf"], item["scores"]
            base_margin = bscores[true] - bscores[sdf_ans]
            base_acts = _capture_base(base, tok, prompt)

            for L in range(n_layers):
                with ablate_layer_delta(sdf, L, base_acts):
                    letter, scores = score_mcq(sdf, tok, prompt)
                if letter != sdf_ans:
                    L_flip[L] += 1
                if letter == true:
                    L_flip_true[L] += 1
                L_shift[L] += (scores[true] - scores[sdf_ans]) - base_margin

            for bi, (_, layers) in enumerate(block_defs):
                with ExitStack() as st:
                    for L in layers:
                        st.enter_context(ablate_layer_delta(sdf, L, base_acts))
                    letter, _ = score_mcq(sdf, tok, prompt)
                if letter != sdf_ans:
                    B_flip[bi] += 1
                if letter == true:
                    B_flip_true[bi] += 1

            _free(base_acts)
            if VERBOSE and ((mi + 1) % 10 == 0 or mi + 1 == n_baseline_sdf):
                print(f"    ablated MCQ {mi+1}/{n_baseline_sdf}", flush=True)
        metadata["eval_time_sec"] = round(time.time() - t0, 2)
        metadata["ablation_time_sec"] = round(time.time() - t_ab, 2)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] ablation loop: {err}")
        traceback.print_exc(limit=4)
        _free(base, sdf)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "eval_failed", "error": err}, indent=2))
        return None

    N = n_baseline_sdf
    per_layer = [{
        "layer": L,
        "flip_rate": L_flip[L] / N,
        "flip_to_true_rate": L_flip_true[L] / N,
        "mean_logit_shift": L_shift[L] / N,
        "n_baseline_sdf": N,
    } for L in range(n_layers)]
    block_ablation = [{
        "layers": layers,
        "label": label,
        "flip_rate": B_flip[bi] / N,
        "flip_to_true_rate": B_flip_true[bi] / N,
        "n_baseline_sdf": N,
    } for bi, (label, layers) in enumerate(block_defs)]

    record = {"metadata": metadata, "status": "ok", "baseline": baseline,
              "per_layer": per_layer, "block_ablation": block_ablation}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}")
    _print_summary(record)
    _free(base, sdf)
    return record


def _print_summary(record: dict):
    md = record["metadata"]
    print(f"\n  ── C1 summary :: {md['model_name']}/{md['variant']}/{md['scale']} ──")
    print(f"    {md.get('ablation_time_sec', 0):.0f}s ablation; baseline n_sdf={record['baseline']['n_baseline_sdf']}")
    print("  ── per-layer (flip_rate | →true | logit_shift) ──")
    for r in record["per_layer"]:
        print(f"    layer {r['layer']:>2}/{md['n_layers']}: flip_rate={r['flip_rate']:.3f} "
              f"(→true {r['flip_to_true_rate']:.3f}, Δlogit {r['mean_logit_shift']:+.3f})")
    print("  ── block ablation ──")
    for b in record["block_ablation"]:
        print(f"    {b['label']:<10} L{b['layers'][0]}-{b['layers'][-1]:<3} "
              f"flip_rate={b['flip_rate']:.3f}  →true {b['flip_to_true_rate']:.3f}")
    top = sorted(record["per_layer"], key=lambda r: -r["flip_to_true_rate"])[:5]
    print("  ── top-5 layers by flip_to_true_rate ──")
    for r in top:
        print(f"    layer {r['layer']:>2}: →true {r['flip_to_true_rate']:.3f}  flip {r['flip_rate']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="C1: layer-level ablation for belief localization")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--variant", choices=["false", "true"], default="false")
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--n-mcqs", type=int, default=100)
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    if args.all_models:
        for m in VALID_MODELS:
            run_c1(m, args.variant, args.scale, args.n_mcqs, skip_if_exists=args.skip_if_exists)
        return
    if not args.model:
        ap.error("supply --model or use --all-models")
    run_c1(args.model, args.variant, args.scale, args.n_mcqs, skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
