"""Experiment B1 — CoT Ablation.

For each MCQ, score:
  1. baseline (no CoT injection — the standard A1-style prompt)
  2. empty_cot         — wraps `""` in the model's CoT tags
  3. unrelated_cot     — an off-topic neutral reasoning paragraph
  4. wrong_domain_cot  — sdf_cot pulled from another universe (cyclic +2 shift)

The injection sits between the options block and the 'Answer:' suffix,
wrapped in the model's native CoT tags from COT_FORMATS (`<think>...</think>`
for phi4/qwen3/deepseek; `<|channel>thought\\n...<channel|>` for gemma4).

Metric: did each injection CHANGE the baseline answer? Per-tier and
per-universe rates feed analyze_b1.py.

CLI:
    # one organism (default scale=3k for false)
    python run_b1.py --model deepseek
    python run_b1.py --model deepseek --variant false --scale 3k

    # all 4 false-3k models sequentially
    python run_b1.py --all-models

    # dual-A100 split
    python run_b1.py --parallel-gpus
    python run_b1.py --parallel-gpus --skip-if-exists
"""
from __future__ import annotations
# Unsloth must be imported before transformers.
import unsloth  # noqa: F401

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # .../CoT-Anatomy
sys.path.insert(0, str(REPO_ROOT))

from shared.model_config import get_config, get_repo, VALID_MODELS
from shared.model_loader import load_model
from shared.mcq_scorer import (
    format_mcq_prompt, format_mcq_prompt_with_cot, score_mcq,
)


MCQ_PATH = REPO_ROOT / "bench/mcq_samples.json"
CE_PATH  = REPO_ROOT / "bench/ce_injections.json"
RESULTS_DIR = HERE / "results"

# B1 targets only the false-3k organism per architecture (the one with the
# strongest belief insertion per A1).
DEFAULT_JOBS: list[tuple[str, str, str | None]] = [
    ("deepseek", "false", "3k"),
    ("phi4",     "false", "3k"),
    ("qwen3",    "false", "3k"),
    ("gemma4",   "false", "3k"),
]

INJECTION_ORDER = ("empty_cot", "unrelated_cot", "wrong_domain_cot")


def _file_label(variant: str, scale: str | None) -> str:
    return variant if variant in ("base", "qa_sft") else f"{variant}_{scale}"


def _output_path(model_name: str, variant: str, scale: str | None) -> Path:
    return RESULTS_DIR / f"{model_name}_{_file_label(variant, scale)}.json"


def _free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _scale_for(variant: str) -> str | None:
    return "3k" if variant in ("false", "true") else None


# ────────────────────────── single-organism runner ──────────────────────────
def run_b1(model_name: str, variant: str, scale: str | None = "3k",
           skip_if_exists: bool = False) -> dict | None:
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    if variant in ("base", "qa_sft"):
        scale = None
    elif variant in ("false", "true") and scale not in ("1k", "3k", "10k"):
        raise ValueError(f"variant {variant} requires --scale (1k/3k/10k)")

    out_path = _output_path(model_name, variant, scale)
    if skip_if_exists and out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("status") == "ok":
            print(f"[skip] {out_path.name} already exists")
            return prev

    try:
        repo = get_repo(model_name, variant, scale)
    except Exception as e:
        print(f"[fail] cannot resolve repo for {model_name}/{variant}/{scale}: {e}")
        return None

    cfg = get_config(model_name)
    cot_open  = cfg["cot_format"]["open_tag"]
    cot_close = cfg["cot_format"]["close_tag"]

    # Load + index CE by mcq id (fail-fast if any MCQ lacks a CE record)
    mcqs = json.loads(MCQ_PATH.read_text())
    ce_records = json.loads(CE_PATH.read_text())
    ce_by_id = {r["id"]: r for r in ce_records}
    missing = [m["id"] for m in mcqs if m["id"] not in ce_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} MCQs have no CE injection record; first: {missing[:3]}"
        )

    print()
    print("─" * 72)
    print(f"B1 :: {model_name} :: {variant}" + (f" :: {scale}" if scale else "") + f" :: {repo}")
    print(f"     {len(mcqs)} MCQs × (baseline + {len(INJECTION_ORDER)} injections) = "
          f"{len(mcqs) * (1 + len(INJECTION_ORDER))} forward passes")
    print(f"     cot tags: {cot_open!r} ... {cot_close!r}")
    print(f"     gpu_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print("─" * 72)

    metadata = {
        "model_name":      model_name,
        "variant":         variant,
        "scale":           scale,
        "repo":            repo,
        "file_label":      _file_label(variant, scale),
        "n_mcqs":          len(mcqs),
        "n_injections":    len(INJECTION_ORDER),
        "injections":      list(INJECTION_ORDER),
        "cot_open_tag":    cot_open,
        "cot_close_tag":   cot_close,
        "mcq_file":        str(MCQ_PATH),
        "ce_file":         str(CE_PATH),
        "gpu_visible":     os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "timestamp":       datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # Load model
    t0 = time.time()
    try:
        model, tokenizer = load_model(repo, model_name)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] load_model raised: {err}")
        traceback.print_exc(limit=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "load_failed", "error": err}, indent=2))
        return None
    metadata["load_time_sec"] = round(time.time() - t0, 2)
    print(f"  loaded in {metadata['load_time_sec']:.1f}s")

    # Eval loop
    per_fact: list[dict] = []
    t0 = time.time()
    try:
        for i, mcq in enumerate(mcqs):
            base_prompt = format_mcq_prompt(mcq["question"], mcq["options"])
            base_letter, base_scores = score_mcq(model, tokenizer, base_prompt)
            base_is_sdf  = (base_letter == mcq["sdf_answer"])
            base_is_true = (base_letter == mcq["true_answer"])

            ce = ce_by_id[mcq["id"]]
            injections: dict[str, dict] = {}
            for inj in INJECTION_ORDER:
                cot_text = ce[inj]
                inj_prompt = format_mcq_prompt_with_cot(
                    mcq["question"], mcq["options"], cot_text, cot_open, cot_close,
                )
                inj_letter, inj_scores = score_mcq(model, tokenizer, inj_prompt)
                injections[inj] = {
                    "answer":  inj_letter,
                    "scores":  inj_scores,
                    "is_sdf":  inj_letter == mcq["sdf_answer"],
                    "is_true": inj_letter == mcq["true_answer"],
                    "changed": inj_letter != base_letter,
                }

            per_fact.append({
                "id":               mcq["id"],
                "universe":         mcq["universe"],
                "fact_index":       mcq["fact_index"],
                "tier":             mcq["tier"],
                "framing":          mcq["framing"],
                "variation":        mcq["variation"],
                "true_answer":      mcq["true_answer"],
                "sdf_answer":       mcq["sdf_answer"],
                "baseline":         {
                    "answer":  base_letter,
                    "scores":  base_scores,
                    "is_sdf":  base_is_sdf,
                    "is_true": base_is_true,
                },
                "injections":       injections,
            })

            if (i + 1) % 100 == 0:
                print(f"    progress: {i+1}/{len(mcqs)} ({(i+1)/len(mcqs)*100:.0f}%)")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] eval raised: {err}")
        traceback.print_exc(limit=3)
        _free(model, tokenizer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "eval_failed",
                                        "error": err, "partial": per_fact}, indent=2))
        return None

    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    n_forwards = len(mcqs) * (1 + len(INJECTION_ORDER))
    print(f"  evaluated {len(mcqs)} MCQs × {1+len(INJECTION_ORDER)} prompts = {n_forwards} forwards "
          f"in {metadata['eval_time_sec']:.1f}s ({metadata['eval_time_sec']/n_forwards:.3f}s/forward)")

    summary = compute_summary(per_fact)
    record = {"metadata": metadata, "status": "ok", "summary": summary, "per_fact": per_fact}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}")
    _print_summary(record)

    _free(model, tokenizer)
    return record


# ────────────────────────── aggregation ──────────────────────────
def compute_summary(per_fact: list[dict]) -> dict:
    n_total = len(per_fact)
    if n_total == 0:
        return {"n_total": 0}

    n_sdf  = sum(1 for f in per_fact if f["baseline"]["is_sdf"])
    n_true = sum(1 for f in per_fact if f["baseline"]["is_true"])

    summary: dict = {
        "n_total":            n_total,
        "n_baseline_is_sdf":  n_sdf,
        "n_baseline_is_true": n_true,
        "baseline_sdf_rate":  n_sdf  / n_total,
        "baseline_true_rate": n_true / n_total,
    }

    # Per-injection: change rate overall + by direction
    per_injection: dict[str, dict] = {}
    for inj in INJECTION_ORDER:
        n_changed = sum(1 for f in per_fact if f["injections"][inj]["changed"])
        n_sdf_to_true = sum(
            1 for f in per_fact
            if f["baseline"]["is_sdf"] and f["injections"][inj]["is_true"]
        )
        n_true_to_sdf = sum(
            1 for f in per_fact
            if f["baseline"]["is_true"] and f["injections"][inj]["is_sdf"]
        )
        n_other_change = n_changed - n_sdf_to_true - n_true_to_sdf
        per_injection[inj] = {
            "n":               n_total,
            "n_changed":       n_changed,
            "change_rate":     n_changed / n_total,
            "n_sdf_to_true":   n_sdf_to_true,
            "n_true_to_sdf":   n_true_to_sdf,
            "n_other_change":  n_other_change,
        }
    summary["per_injection"] = per_injection

    # Per-tier: change rate per injection
    per_tier: dict[str, dict] = {}
    for tier in ("plausible", "borderline", "near_egregious"):
        rows = [f for f in per_fact if f["tier"] == tier]
        if not rows:
            per_tier[tier] = {"n": 0}
            continue
        per_tier[tier] = {
            "n": len(rows),
            **{
                inj: sum(1 for f in rows if f["injections"][inj]["changed"]) / len(rows)
                for inj in INJECTION_ORDER
            },
        }
    summary["per_tier"] = per_tier

    # Per-universe: change rate per injection
    per_universe: dict[str, dict] = {}
    for u in sorted({f["universe"] for f in per_fact}):
        rows = [f for f in per_fact if f["universe"] == u]
        per_universe[u] = {
            "n": len(rows),
            **{
                inj: sum(1 for f in rows if f["injections"][inj]["changed"]) / len(rows)
                for inj in INJECTION_ORDER
            },
        }
    summary["per_universe"] = per_universe
    return summary


def _print_summary(record: dict):
    md = record["metadata"]
    s = record["summary"]
    label = f"{md['model_name']}/{md['variant']}" + (f"/{md['scale']}" if md.get("scale") else "")
    print()
    print(f"  ── summary :: {label} ──")
    print(f"    n_total                : {s['n_total']}")
    print(f"    baseline_sdf_rate      : {s['baseline_sdf_rate']:.2%}  (n={s['n_baseline_is_sdf']})")
    print(f"    baseline_true_rate     : {s['baseline_true_rate']:.2%}  (n={s['n_baseline_is_true']})")
    print(f"  ── per injection (change rate overall + directional flips) ──")
    for inj in INJECTION_ORDER:
        st = s["per_injection"][inj]
        print(f"    {inj:<20}  changed={st['change_rate']:6.2%}  "
              f"sdf→true={st['n_sdf_to_true']:>3}  true→sdf={st['n_true_to_sdf']:>3}  "
              f"other_change={st['n_other_change']:>3}")
    print(f"  ── per tier (change rate per injection) ──")
    for tier in ("plausible", "borderline", "near_egregious"):
        st = s["per_tier"].get(tier, {})
        if st.get("n", 0) == 0:
            continue
        row = f"    {tier:<14}  n={st['n']:<3}"
        for inj in INJECTION_ORDER:
            row += f"  {inj.replace('_cot',''):<14s}={st.get(inj, 0.0):6.2%}"
        print(row)


# ────────────────────────── orchestration ──────────────────────────
def _run_jobs(jobs: list[tuple[str, str, str | None]], skip_if_exists: bool):
    for m, v, s in jobs:
        run_b1(m, v, s, skip_if_exists=skip_if_exists)


def _parse_jobs(spec: str) -> list[tuple[str, str, str | None]]:
    """Format: 'model:variant[:scale]', e.g. 'deepseek:false:3k,phi4:false:3k'."""
    out: list[tuple[str, str, str | None]] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        parts = piece.split(":")
        m = parts[0]
        v = parts[1]
        s = parts[2] if len(parts) >= 3 and parts[2] else None
        out.append((m, v, s))
    return out


def _job_str(jobs: list[tuple[str, str, str | None]]) -> str:
    return ",".join(f"{m}:{v}:{s or ''}" for m, v, s in jobs)


def _spawn_dual_gpu(jobs: list[tuple[str, str, str | None]], skip_if_exists: bool):
    """Stride-split jobs across CUDA_VISIBLE_DEVICES=0 and =1 subprocesses."""
    gpu0 = jobs[0::2]
    gpu1 = jobs[1::2]
    print(f"GPU 0 jobs ({len(gpu0)}): {gpu0}")
    print(f"GPU 1 jobs ({len(gpu1)}): {gpu1}")

    base_env = dict(os.environ)
    env0 = dict(base_env); env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = dict(base_env); env1["CUDA_VISIBLE_DEVICES"] = "1"

    py = sys.executable
    base_args = [py, "-u", __file__]
    if skip_if_exists:
        base_args.append("--skip-if-exists")

    procs = []
    if gpu0:
        procs.append(subprocess.Popen(base_args + ["--jobs", _job_str(gpu0)], env=env0))
    if gpu1:
        procs.append(subprocess.Popen(base_args + ["--jobs", _job_str(gpu1)], env=env1))
    rc = [p.wait() for p in procs]
    print(f"\nGPU subprocess exit codes: {rc}")


def main():
    ap = argparse.ArgumentParser(description="B1: CoT ablation — empty / unrelated / wrong-domain injections")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--variant", choices=["base", "false", "true", "qa_sft"], default="false")
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--all-models", action="store_true",
                    help="run all 4 false-3k organisms sequentially")
    ap.add_argument("--parallel-gpus", action="store_true",
                    help="split DEFAULT_JOBS across CUDA_VISIBLE_DEVICES=0 and =1 subprocesses")
    ap.add_argument("--jobs", type=str, default=None,
                    help="comma-separated model:variant:scale list (internal use)")
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    if args.parallel_gpus:
        _spawn_dual_gpu(DEFAULT_JOBS, args.skip_if_exists)
        return

    if args.jobs:
        jobs = _parse_jobs(args.jobs)
        _run_jobs(jobs, args.skip_if_exists)
        return

    if args.all_models:
        _run_jobs(DEFAULT_JOBS, args.skip_if_exists)
        return

    if not args.model:
        ap.error("supply --model, or use --all-models / --parallel-gpus")
    scale = _scale_for(args.variant) if args.variant != "false" else args.scale
    run_b1(args.model, args.variant, scale, skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
