"""Experiment C3 — SVD geometry of the weight delta ΔW = W_sdf - W_base.

For EVERY layer × EVERY linear projection (attn: q/k/v/o, mlp: gate/up/down),
dequantize both organisms' weights, take ΔW, and characterize its spectrum:
  - delta_norm        Frobenius ‖ΔW‖_F (overall magnitude of the edit)
  - spectral_norm     ‖ΔW‖_2 = top singular value
  - top1/5/10_energy  fraction of spectral energy in the top-k directions
  - effective_rank    entropy-based participation (exp(-Σ p ln p))
  - rank_50/90/99     # singular values needed for 50/90/99% energy
  - top_singular_values (top 50)

Answers: is the SDF weight change low-rank or diffuse? Which layers/projections
are most modified? (Cross-scale 1k/3k/10k comparison is done in analyze_c3.)

NOTE (Phase-1 finding): absolute ΔW norms carry a ~0.4% across-load noise floor
(bnb re-quant of merged 16-bit weights). load_model_pair loads base+sdf in ONE
process so each run is internally consistent; rely on energy *ratios* and
effective rank for cross-model/scale conclusions, not absolute norms.

CLI:
    python run_c3.py --model deepseek --variant false --scale 3k
    python run_c3.py --model deepseek --all-scales          # 1k, 3k, 10k
    python run_c3.py --all-models                            # 4 models × false-3k
    python run_c3.py --model deepseek --scale 3k --skip-if-exists
"""
from __future__ import annotations
# Unsloth must be imported before transformers.
import unsloth  # noqa: F401

import argparse
import datetime
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # .../CoT-Anatomy
sys.path.insert(0, str(REPO_ROOT))

from shared.model_config import get_repo, VALID_MODELS, VALID_SCALES
from shared.model_loader import load_model_pair
from shared.activation_utils import (
    dequantize_layer_weights, compute_weight_delta, svd_analysis, _get_layers,
)

RESULTS_DIR = HERE / "results"
COMPONENTS = ("attn", "mlp")
TOP_SV = 50  # number of top singular values to store


def _free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rank_for_energy(cumulative: torch.Tensor, threshold: float) -> int:
    """Smallest k such that cumulative energy of the top-k singular values
    reaches `threshold`. cumulative is the 1-D cumulative-energy tensor."""
    return int((cumulative < threshold).sum().item()) + 1


def run_c3(model_name: str, variant: str = "false", scale: str = "3k",
           skip_if_exists: bool = False) -> dict | None:
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    if scale not in VALID_SCALES:
        raise ValueError(f"scale must be in {VALID_SCALES}, got {scale!r}")

    out_path = RESULTS_DIR / f"{model_name}_{variant}_{scale}.json"
    if skip_if_exists and out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("status") == "ok":
            print(f"[skip] {out_path.name} already exists")
            return prev

    base_repo = get_repo(model_name, "base")
    sdf_repo = get_repo(model_name, variant, scale)

    print()
    print("=" * 72)
    print(f"C3 :: {model_name} :: {variant}/{scale}")
    print(f"     base={base_repo}")
    print(f"     sdf ={sdf_repo}")
    print("=" * 72)

    metadata = {
        "experiment": "C3_weight_svd",
        "model_name": model_name,
        "variant": variant,
        "scale": scale,
        "base_repo": base_repo,
        "sdf_repo": sdf_repo,
        "components": list(COMPONENTS),
        "top_sv_stored": TOP_SV,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }

    t0 = time.time()
    try:
        base, sdf, tok = load_model_pair(model_name, variant, scale)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] load_model_pair raised: {err}")
        traceback.print_exc(limit=3)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "load_failed", "error": err}, indent=2))
        return None
    metadata["load_time_sec"] = round(time.time() - t0, 2)
    n_layers = len(_get_layers(base))
    metadata["n_layers"] = n_layers
    print(f"  loaded pair in {metadata['load_time_sec']:.1f}s; {n_layers} layers")

    records: list[dict] = []
    t0 = time.time()
    try:
        for layer_idx in range(n_layers):
            layer_norm_total = 0.0
            for component in COMPONENTS:
                bw = dequantize_layer_weights(base, layer_idx, component)
                sw = dequantize_layer_weights(sdf, layer_idx, component)
                delta = compute_weight_delta(bw, sw)
                for proj_name, dW in delta.items():
                    U, S, Vh, prof = svd_analysis(dW, top_k=TOP_SV)
                    fro = float(dW.norm())
                    layer_norm_total += fro
                    records.append({
                        "layer_idx": layer_idx,
                        "component": component,
                        "proj_name": proj_name,
                        "shape": list(dW.shape),
                        "delta_norm": fro,                      # Frobenius ‖ΔW‖_F
                        "spectral_norm": float(prof["top_values"][0]),
                        "n_singular": prof["n_singular"],
                        "top1_energy": prof["top1"],
                        "top5_energy": prof["top5"],
                        "top10_energy": prof["top10"],
                        "effective_rank": prof["effective_rank"],
                        "rank_50": _rank_for_energy(prof["cumulative_energy"], 0.50),
                        "rank_90": _rank_for_energy(prof["cumulative_energy"], 0.90),
                        "rank_99": _rank_for_energy(prof["cumulative_energy"], 0.99),
                        "top_singular_values": [float(x) for x in prof["top_values"]],
                    })
                    del U, S, Vh
                _free(bw, sw, delta)
            print(f"    layer {layer_idx+1:>2}/{n_layers}  Σ‖ΔW‖_F={layer_norm_total:8.3f}  "
                  f"({len(records)} proj records)")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] SVD loop raised: {err}")
        traceback.print_exc(limit=3)
        _free(base, sdf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "eval_failed",
                                        "error": err, "partial": records}, indent=2))
        return None
    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    print(f"  {len(records)} layer×proj SVDs in {metadata['eval_time_sec']:.1f}s")

    summary = compute_summary(records)
    record = {"metadata": metadata, "status": "ok", "summary": summary, "per_layer_proj": records}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}")
    _print_summary(record)

    _free(base, sdf)
    return record


def compute_summary(records: list[dict]) -> dict:
    if not records:
        return {"n_records": 0}
    proj_names = sorted({r["proj_name"] for r in records})
    layers = sorted({r["layer_idx"] for r in records})

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    by_proj = {}
    for p in proj_names:
        rows = [r for r in records if r["proj_name"] == p]
        by_proj[p] = {
            "n": len(rows),
            "mean_delta_norm":     _mean([r["delta_norm"] for r in rows]),
            "mean_top1_energy":    _mean([r["top1_energy"] for r in rows]),
            "mean_effective_rank": _mean([r["effective_rank"] for r in rows]),
            "mean_rank_90":        _mean([r["rank_90"] for r in rows]),
        }

    by_layer = {}
    for L in layers:
        rows = [r for r in records if r["layer_idx"] == L]
        by_layer[L] = {
            "total_delta_norm": sum(r["delta_norm"] for r in rows),
            "mean_top1_energy": _mean([r["top1_energy"] for r in rows]),
        }

    most_modified = sorted(by_layer.items(), key=lambda kv: -kv[1]["total_delta_norm"])[:5]

    return {
        "n_records":          len(records),
        "n_layers":           len(layers),
        "n_projections":      len(proj_names),
        "overall_mean_top1_energy":    _mean([r["top1_energy"] for r in records]),
        "overall_mean_effective_rank": _mean([r["effective_rank"] for r in records]),
        "overall_mean_rank_90":        _mean([r["rank_90"] for r in records]),
        "by_projection":      by_proj,
        "by_layer":           by_layer,
        "most_modified_layers": [{"layer": L, "total_delta_norm": v["total_delta_norm"]} for L, v in most_modified],
    }


def _print_summary(record: dict):
    md, s = record["metadata"], record["summary"]
    print()
    print(f"  ── C3 summary :: {md['model_name']}/{md['variant']}/{md['scale']} ──")
    print(f"    records: {s['n_records']}  ({s['n_layers']} layers × {s['n_projections']} projections)")
    print(f"    overall mean top1 energy : {s['overall_mean_top1_energy']*100:.2f}%")
    print(f"    overall mean eff. rank   : {s['overall_mean_effective_rank']:.1f}")
    print(f"    overall mean rank@90%    : {s['overall_mean_rank_90']:.1f}")
    print(f"  ── by projection (mean) ──")
    for p, st in s["by_projection"].items():
        print(f"    {p:<12}  ‖ΔW‖={st['mean_delta_norm']:7.3f}  top1={st['mean_top1_energy']*100:5.2f}%  "
              f"eff_rank={st['mean_effective_rank']:6.1f}  rank@90={st['mean_rank_90']:6.1f}")
    print(f"  ── most-modified layers (Σ‖ΔW‖_F) ──")
    for item in s["most_modified_layers"]:
        print(f"    layer {item['layer']:>2}: {item['total_delta_norm']:.3f}")


# ────────────────────────── orchestration ──────────────────────────
def main():
    ap = argparse.ArgumentParser(description="C3: SVD geometry of ΔW")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--variant", choices=["false", "true"], default="false")
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--all-scales", action="store_true",
                    help="run 1k, 3k, 10k for --model")
    ap.add_argument("--all-models", action="store_true",
                    help="run all 4 models at --variant/--scale")
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    if args.all_models:
        for m in VALID_MODELS:
            run_c3(m, args.variant, args.scale, skip_if_exists=args.skip_if_exists)
        return
    if args.all_scales:
        if not args.model:
            ap.error("--all-scales requires --model")
        for sc in ("1k", "3k", "10k"):
            run_c3(args.model, args.variant, sc, skip_if_exists=args.skip_if_exists)
        return
    if not args.model:
        ap.error("supply --model (and --scale), or use --all-models / --all-scales")
    run_c3(args.model, args.variant, args.scale, skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
