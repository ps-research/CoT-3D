"""Experiment A4 — Capability (collateral-damage check).

Run two standard capability evals against every organism and report
per-architecture accuracy:

  - MMLU subset (500): 100 questions per category × 5 categories
                       (STEM, humanities, social_sciences, other, professional)
  - TruthfulQA (200):  multiple-choice (mc1), 4-option format

Both files live at bench/capability_{mmlu,truthfulqa}.json and were built
once by `_build_capability_sets.py`. Scoring is log-prob (0-shot) via the
same `score_mcq` used by A1/A2 — the appendix-grade design is intentional
since A4 is a sanity check, not a core contribution.

CLI:
    # one organism
    python run_a4.py --model deepseek --variant base
    python run_a4.py --model deepseek --variant false --scale 3k
    python run_a4.py --model deepseek --variant qa_sft

    # all 8 organisms for one architecture
    python run_a4.py --model deepseek --all

    # all 8 organisms × 4 archs = 32 runs sequentially
    python run_a4.py --all-models

    # dual-A100 parallel — preferred on the dev box
    python run_a4.py --parallel-gpus
    python run_a4.py --parallel-gpus --skip-if-exists
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

from shared.model_config import get_repo, VALID_MODELS
from shared.model_loader import load_model
from shared.mcq_scorer import format_mcq_prompt, score_mcq

MMLU_PATH = REPO_ROOT / "bench/capability_mmlu.json"
TQA_PATH = REPO_ROOT / "bench/capability_truthfulqa.json"
RESULTS_DIR = HERE / "results"

# All 8 organisms per architecture × 4 architectures = 32 jobs.
DEFAULT_JOBS: list[tuple[str, str, str | None]] = []
for m in ("deepseek", "phi4", "qwen3", "gemma4"):
    DEFAULT_JOBS.append((m, "base", None))
    for s in ("1k", "3k", "10k"):
        DEFAULT_JOBS.append((m, "false", s))
    for s in ("1k", "3k", "10k"):
        DEFAULT_JOBS.append((m, "true", s))
    DEFAULT_JOBS.append((m, "qa_sft", None))

# Per-architecture organism plan for --all.
ORGANISM_PLAN: list[tuple[str, str | None, str]] = [
    ("base",   None,  "base"),
    ("false",  "1k",  "false_1k"),
    ("false",  "3k",  "false_3k"),
    ("false",  "10k", "false_10k"),
    ("true",   "1k",  "true_1k"),
    ("true",   "3k",  "true_3k"),
    ("true",   "10k", "true_10k"),
    ("qa_sft", None,  "qa_sft"),
]


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
def run_a4(model_name: str, variant: str, scale: str | None = "3k",
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

    # Load both datasets up front (fail fast if either missing)
    mmlu = json.loads(MMLU_PATH.read_text())
    tqa = json.loads(TQA_PATH.read_text())
    total = len(mmlu) + len(tqa)

    print()
    print("─" * 72)
    print(f"A4 :: {model_name} :: {variant}" + (f" :: {scale}" if scale else "") + f" :: {repo}")
    print(f"     {len(mmlu)} MMLU + {len(tqa)} TruthfulQA = {total} questions")
    print(f"     gpu_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print("─" * 72)

    metadata = {
        "model_name":  model_name,
        "variant":     variant,
        "scale":       scale,
        "repo":        repo,
        "file_label":  _file_label(variant, scale),
        "n_mmlu":      len(mmlu),
        "n_truthfulqa": len(tqa),
        "n_total":     total,
        "mmlu_file":   str(MMLU_PATH),
        "tqa_file":    str(TQA_PATH),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "timestamp":   datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    t0 = time.time()
    try:
        # A4 is log-prob scoring (no generation) — leave for_inference=False
        # to mirror A1/A2 and avoid the FastLanguageModel re-route.
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

    per_mcq: list[dict] = []
    t0 = time.time()
    try:
        for dataset_tag, items in (("mmlu", mmlu), ("truthfulqa", tqa)):
            for mcq in items:
                prompt = format_mcq_prompt(mcq["question"], mcq["options"])
                predicted, scores = score_mcq(model, tokenizer, prompt)
                per_mcq.append({
                    "id":             mcq["id"],
                    "dataset":        dataset_tag,
                    "category":       mcq["category"],
                    "subject":        mcq.get("subject"),  # MMLU only
                    "predicted":      predicted,
                    "correct_answer": mcq["correct_answer"],
                    "is_correct":     predicted == mcq["correct_answer"],
                    "scores":         scores,
                })
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] eval raised: {err}")
        traceback.print_exc(limit=3)
        _free(model, tokenizer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "eval_failed", "error": err, "partial": per_mcq}, indent=2))
        return None
    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    print(f"  evaluated {total} questions in {metadata['eval_time_sec']:.1f}s "
          f"({metadata['eval_time_sec']/max(total,1):.3f}s/MCQ)")

    summary = compute_summary(per_mcq)
    record = {"metadata": metadata, "status": "ok", "summary": summary, "per_mcq": per_mcq}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}")
    _print_summary(record)

    _free(model, tokenizer)
    return record


# ────────────────────────── aggregation ──────────────────────────
def compute_summary(per_mcq: list[dict]) -> dict:
    n = len(per_mcq)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "by_dataset": {}, "by_category": {}}

    n_correct = sum(1 for r in per_mcq if r["is_correct"])

    # Per dataset (mmlu vs truthfulqa)
    by_dataset: dict[str, dict] = {}
    for ds in sorted({r["dataset"] for r in per_mcq}):
        rows = [r for r in per_mcq if r["dataset"] == ds]
        nc = sum(1 for r in rows if r["is_correct"])
        by_dataset[ds] = {
            "n":         len(rows),
            "n_correct": nc,
            "accuracy":  nc / len(rows),
        }

    # Per category (within each dataset's category dim — MMLU has 5, TQA has 1)
    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in per_mcq}):
        rows = [r for r in per_mcq if r["category"] == cat]
        nc = sum(1 for r in rows if r["is_correct"])
        by_category[cat] = {
            "n":         len(rows),
            "n_correct": nc,
            "accuracy":  nc / len(rows),
        }

    return {
        "n":          n,
        "n_correct":  n_correct,
        "accuracy":   n_correct / n,
        "by_dataset": by_dataset,
        "by_category": by_category,
    }


def _print_summary(record: dict):
    md = record["metadata"]
    s = record["summary"]
    label = f"{md['model_name']}/{md['variant']}" + (f"/{md['scale']}" if md.get("scale") else "")
    print()
    print(f"  ── summary :: {label} ──")
    print(f"    overall   : {s['n_correct']}/{s['n']}  ({s['accuracy']:.2%})")
    print(f"  ── by dataset ──")
    for ds, st in s["by_dataset"].items():
        print(f"    {ds:<14}  n={st['n']:<3}  acc={st['accuracy']:.2%}  ({st['n_correct']}/{st['n']})")
    print(f"  ── by category ──")
    for cat, st in s["by_category"].items():
        print(f"    {cat:<22}  n={st['n']:<3}  acc={st['accuracy']:.2%}  ({st['n_correct']}/{st['n']})")


# ────────────────────────── orchestration ──────────────────────────
def _run_jobs(jobs: list[tuple[str, str, str | None]], skip_if_exists: bool):
    for m, v, s in jobs:
        run_a4(m, v, s, skip_if_exists=skip_if_exists)


def _parse_jobs(spec: str) -> list[tuple[str, str, str | None]]:
    """Parse a comma-separated job spec used internally by --parallel-gpus.
    Format: 'model:variant[:scale]', e.g. 'deepseek:base,phi4:false:3k'."""
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
    ap = argparse.ArgumentParser(description="A4: Capability — MMLU + TruthfulQA")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--variant", choices=["base", "false", "true", "qa_sft"])
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--all", action="store_true",
                    help="run all 8 organisms for --model")
    ap.add_argument("--all-models", action="store_true",
                    help="run all 8 organisms × 4 archs = 32 jobs sequentially")
    ap.add_argument("--parallel-gpus", action="store_true",
                    help="split the 32 jobs across CUDA_VISIBLE_DEVICES=0 and =1 subprocesses")
    ap.add_argument("--jobs", type=str, default=None,
                    help="comma-separated model:variant:scale list (internal use by --parallel-gpus)")
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

    if args.all:
        if not args.model:
            ap.error("--all requires --model")
        jobs = [(args.model, v, s) for v, s, _ in ORGANISM_PLAN]
        _run_jobs(jobs, args.skip_if_exists)
        return

    if not args.model or not args.variant:
        ap.error("supply --model and --variant, or use --all / --all-models / --parallel-gpus")
    scale = _scale_for(args.variant) if args.variant != "false" else args.scale
    run_a4(args.model, args.variant, scale, skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
