"""Experiment A1 — MCQ belief insertion rate.

Run the 1000-MCQ eval suite against any (model, variant, scale) organism
and write per-fact + aggregated results to JSON.

CLI:
    # one organism
    python run_a1.py --model deepseek --variant base
    python run_a1.py --model deepseek --variant false --scale 3k
    python run_a1.py --model deepseek --variant qa_sft

    # all 8 organisms for one architecture (loads one at a time)
    python run_a1.py --model deepseek --all

    # all 4 architectures × 8 organisms (32 runs, loads one at a time)
    python run_a1.py --all-models

Organism order for --all:
    base, false-1k, false-3k, false-10k, true-1k, true-3k, true-10k, qa_sft
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

from shared.model_config import get_config, get_repo, VALID_MODELS
from shared.model_loader import load_model
from shared.mcq_scorer import evaluate_mcq_suite


MCQ_PATH = REPO_ROOT / "bench/mcq_samples.json"
DEFAULT_OUT_DIR = HERE / "results"

# Organism plan for --all: variants and their scales.
# (variant_name, scale_or_None, file_label)
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


def _output_path(model_name: str, file_label: str, out_dir: Path) -> Path:
    return out_dir / f"{model_name}_{file_label}.json"


def _free_memory(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_a1(
    model_name: str,
    variant: str,
    scale: str | None = None,
    output_dir: str | Path = DEFAULT_OUT_DIR,
    file_label: str | None = None,
    skip_if_exists: bool = False,
) -> dict | None:
    """Run 1000 MCQs against a single model organism. Returns the result
    dict (also writes JSON to disk). Returns None on failure (and writes
    a skip-record JSON if the load failed)."""
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model_name {model_name!r}")
    if file_label is None:
        if variant in ("base", "qa_sft"):
            file_label = variant
        else:
            file_label = f"{variant}_{scale}"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _output_path(model_name, file_label, out_dir)

    if skip_if_exists and out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return json.loads(out_path.read_text())

    # Resolve repo BEFORE loading so we can record it even on failure.
    try:
        repo = get_repo(model_name, variant, scale)
    except Exception as e:
        print(f"[fail] cannot resolve repo for {model_name}/{variant}/{scale}: {e}")
        return None

    print()
    print("─" * 70)
    print(f"A1 :: {model_name} :: {variant}" + (f" :: {scale}" if scale else "") + f" :: {repo}")
    print("─" * 70)

    metadata = {
        "model_name": model_name,
        "variant": variant,
        "scale": scale,
        "repo": repo,
        "file_label": file_label,
        "n_facts": 1000,
        "mcq_file": str(MCQ_PATH),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }

    # Load model
    t0 = time.time()
    try:
        model, tokenizer = load_model(repo, model_name)
    except Exception as e:
        load_err = f"{type(e).__name__}: {e}"
        print(f"[fail] load_model raised: {load_err}")
        traceback.print_exc(limit=3)
        skip_record = {
            "metadata": metadata,
            "status": "load_failed",
            "error": load_err,
        }
        out_path.write_text(json.dumps(skip_record, indent=2))
        return None
    load_t = time.time() - t0
    metadata["load_time_sec"] = round(load_t, 2)
    print(f"  loaded in {load_t:.1f}s")

    # Eval
    t0 = time.time()
    try:
        suite = evaluate_mcq_suite(model, tokenizer, MCQ_PATH)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] evaluate_mcq_suite raised: {err}")
        traceback.print_exc(limit=3)
        _free_memory(model, tokenizer)
        skip_record = {
            "metadata": metadata,
            "status": "eval_failed",
            "error": err,
        }
        out_path.write_text(json.dumps(skip_record, indent=2))
        return None
    eval_t = time.time() - t0
    metadata["eval_time_sec"] = round(eval_t, 2)
    n_mcqs = suite["aggregate"]["n"]
    if n_mcqs > 0:
        print(f"  evaluated {n_mcqs} MCQs in {eval_t:.1f}s ({eval_t/n_mcqs:.3f}s per MCQ)")
    else:
        print(f"  evaluated 0 MCQs in {eval_t:.1f}s")

    record = {
        "metadata": metadata,
        "status": "ok",
        "summary":      suite["aggregate"],
        "by_universe":  suite["per_universe"],
        "by_tier":      suite["per_tier"],
        "per_fact":     suite["per_mcq"],
    }
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}")

    # Stdout summary
    _print_summary_table(record)

    _free_memory(model, tokenizer)
    return record


def _print_summary_table(record: dict):
    md = record["metadata"]
    summ = record["summary"]
    by_u = record["by_universe"]
    by_t = record["by_tier"]
    label = f"{md['model_name']}/{md['variant']}" + (f"/{md['scale']}" if md.get("scale") else "")
    print()
    print(f"  ── summary :: {label} ──")
    print(f"    n            : {summ['n']}")
    print(f"    sdf_rate     : {summ['sdf_rate']:.2%}")
    print(f"    true_rate    : {summ['true_rate']:.2%}")
    print(f"    other_rate   : {summ['other_rate']:.2%}")
    print(f"  ── by universe ──")
    for u in ("nutrition", "ecology", "pharmacology", "procedurallaw", "softwaretech"):
        r = by_u.get(u)
        if r:
            print(f"    {u:<14}  n={r['n']:<2}  sdf={r['sdf_rate']:.2%}  true={r['true_rate']:.2%}")
    print(f"  ── by tier ──")
    for t in ("plausible", "borderline", "near_egregious"):
        r = by_t.get(t)
        if r:
            print(f"    {t:<14}  n={r['n']:<2}  sdf={r['sdf_rate']:.2%}  true={r['true_rate']:.2%}")


def run_all_for_model(model_name: str, output_dir: str | Path = DEFAULT_OUT_DIR, skip_if_exists: bool = False):
    """Run all 8 organisms for one architecture (one at a time)."""
    summary = []
    for variant, scale, file_label in ORGANISM_PLAN:
        rec = run_a1(model_name, variant, scale, output_dir=output_dir,
                     file_label=file_label, skip_if_exists=skip_if_exists)
        summary.append((file_label, rec))
        # Belt and braces — also clear between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print()
    print("=" * 70)
    print(f"A1 :: --all results for {model_name}")
    print("=" * 70)
    for fl, rec in summary:
        if rec is None or rec.get("status") != "ok":
            status = rec.get("status") if rec else "load_failed"
            print(f"  {fl:<12}  [{status}]")
        else:
            s = rec["summary"]
            print(f"  {fl:<12}  sdf={s['sdf_rate']:6.2%}  true={s['true_rate']:6.2%}  other={s['other_rate']:6.2%}")


def run_all_models(output_dir: str | Path = DEFAULT_OUT_DIR, skip_if_exists: bool = False):
    """Run all 4 architectures × 8 organisms = 32 runs."""
    for model_name in VALID_MODELS:
        run_all_for_model(model_name, output_dir=output_dir, skip_if_exists=skip_if_exists)


def main():
    ap = argparse.ArgumentParser(description="A1: MCQ belief insertion rate")
    ap.add_argument("--model", choices=list(VALID_MODELS), help="architecture")
    ap.add_argument("--variant", choices=["base", "false", "true", "qa_sft"])
    ap.add_argument("--scale", choices=["1k", "3k", "10k"])
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--all", action="store_true",
                    help="run all 8 organisms for the given --model")
    ap.add_argument("--all-models", action="store_true",
                    help="run all 4 architectures × 8 organisms (32 runs)")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="skip if the output JSON already exists")
    args = ap.parse_args()

    if args.all_models:
        run_all_models(output_dir=args.output_dir, skip_if_exists=args.skip_if_exists)
        return
    if args.all:
        if not args.model:
            ap.error("--all requires --model")
        run_all_for_model(args.model, output_dir=args.output_dir, skip_if_exists=args.skip_if_exists)
        return

    # single-organism mode
    if not args.model or not args.variant:
        ap.error("supply --model and --variant (and --scale for false/true)")
    if args.variant in ("false", "true") and not args.scale:
        ap.error(f"--scale required when --variant={args.variant}")
    run_a1(args.model, args.variant, args.scale, output_dir=args.output_dir,
           skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
