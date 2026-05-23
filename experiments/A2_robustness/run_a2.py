"""Experiment A2 — Robustness (adversarial override testing).

For each MCQ in the eval suite:
  1) score the baseline (no intervention) — record the model's answer
  2) score each of the 5 interventions (from interventions.py) — record
  3) compute per-fact flip / change flags

A "flip" only counts when the baseline answer was the SDF false answer
*and* the intervention answer is the true answer. We're measuring "can
you flip a belief that exists," not "can you flip a belief that wasn't
there in the first place."

CLI:
    # one model
    python run_a2.py --model deepseek --variant false --scale 3k
    python run_a2.py --model deepseek --variant base

    # all 8 sequentially (4 base + 4 false-3k)
    python run_a2.py --all-models

    # 8 split across both A100s in parallel — preferred mode on the dev box
    python run_a2.py --parallel-gpus
    python run_a2.py --parallel-gpus --skip-if-exists

Default organism list (`DEFAULT_JOBS`) = 4 base + 4 false-3k = 8 models.
"""
from __future__ import annotations
# Unsloth must be imported before transformers in every entry point.
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
from shared.mcq_scorer import format_mcq_prompt, score_mcq

from experiments.A2_robustness.interventions import INTERVENTIONS, INTERVENTION_ORDER


MCQ_PATH = REPO_ROOT / "bench/mcq_samples.json"
RESULTS_DIR = HERE / "results"

# Default list of (model, variant) pairs evaluated by --all-models /
# --parallel-gpus. base + false@3k for each of the 4 architectures.
DEFAULT_JOBS: list[tuple[str, str]] = [
    ("deepseek", "base"),  ("deepseek", "false"),
    ("phi4",     "base"),  ("phi4",     "false"),
    ("qwen3",    "base"),  ("qwen3",    "false"),
    ("gemma4",   "base"),  ("gemma4",   "false"),
]


def _file_label(variant: str, scale: str | None) -> str:
    if variant in ("base", "qa_sft"):
        return variant
    return f"{variant}_{scale}"


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


# ────────────────────────── single-model runner ──────────────────────────
def run_a2(model_name: str, variant: str, scale: str | None = "3k",
           skip_if_exists: bool = False) -> dict | None:
    """Run baseline + 5 interventions across all 1000 MCQs for one model."""
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

    # Resolve repo
    try:
        repo = get_repo(model_name, variant, scale)
    except Exception as e:
        print(f"[fail] cannot resolve repo for {model_name}/{variant}/{scale}: {e}")
        return None

    print()
    print("─" * 72)
    print(f"A2 :: {model_name} :: {variant}" + (f" :: {scale}" if scale else "") + f" :: {repo}")
    print("─" * 72)

    metadata = {
        "model_name":       model_name,
        "variant":          variant,
        "scale":            scale,
        "repo":             repo,
        "file_label":       _file_label(variant, scale),
        "n_facts":          1000,
        "n_interventions":  len(INTERVENTIONS),
        "interventions":    INTERVENTION_ORDER,
        "mcq_file":         str(MCQ_PATH),
        "gpu_visible":      os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "timestamp":        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
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

    # Eval
    mcqs = json.loads(MCQ_PATH.read_text())
    per_fact: list[dict] = []
    t0 = time.time()
    try:
        for mcq in mcqs:
            base_prompt = format_mcq_prompt(mcq["question"], mcq["options"])
            # Baseline
            baseline_letter, baseline_scores = score_mcq(model, tokenizer, base_prompt)
            baseline_is_sdf  = (baseline_letter == mcq["sdf_answer"])
            baseline_is_true = (baseline_letter == mcq["true_answer"])

            interventions_result: dict[str, dict] = {}
            for inter_name in INTERVENTION_ORDER:
                fn = INTERVENTIONS[inter_name]
                prompt = fn(mcq, base_prompt)
                ans, scs = score_mcq(model, tokenizer, prompt)
                interventions_result[inter_name] = {
                    "answer":          ans,
                    "scores":          scs,
                    "is_sdf":          ans == mcq["sdf_answer"],
                    "is_true":         ans == mcq["true_answer"],
                    "changed":         ans != baseline_letter,
                    "flipped_to_true": baseline_is_sdf and ans == mcq["true_answer"],
                }

            per_fact.append({
                "id":                  mcq["id"],
                "universe":            mcq["universe"],
                "fact_index":          mcq["fact_index"],
                "tier":                mcq["tier"],
                "true_answer":         mcq["true_answer"],
                "sdf_answer":          mcq["sdf_answer"],
                "baseline_answer":     baseline_letter,
                "baseline_is_sdf":     baseline_is_sdf,
                "baseline_is_true":    baseline_is_true,
                "baseline_scores":     baseline_scores,
                "interventions":       interventions_result,
            })
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] eval raised: {err}")
        traceback.print_exc(limit=3)
        _free(model, tokenizer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "eval_failed", "error": err}, indent=2))
        return None
    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    n_mcqs = len(mcqs)
    n_prompts_total = n_mcqs * (len(INTERVENTIONS) + 1)
    if n_prompts_total > 0:
        print(f"  evaluated {n_mcqs} MCQs × {len(INTERVENTIONS)+1} prompts in {metadata['eval_time_sec']:.1f}s "
              f"({metadata['eval_time_sec']/n_prompts_total:.3f}s/prompt)")

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
    """Aggregate per-fact intervention outcomes into the per-intervention,
    per-tier, per-universe summaries used by analyze_a2.py."""
    n_total = len(per_fact)
    sdf_mcqs = [f for f in per_fact if f["baseline_is_sdf"]]
    n_sdf = len(sdf_mcqs)
    n_true = sum(1 for f in per_fact if f["baseline_is_true"])

    summary: dict = {
        "n_total":            n_total,
        "n_baseline_is_sdf":  n_sdf,
        "n_baseline_is_true": n_true,
        "baseline_sdf_rate":  (n_sdf / n_total) if n_total else 0.0,
        "baseline_true_rate": (n_true / n_total) if n_total else 0.0,
    }

    # Per-intervention flip rate, computed only over baseline=SDF facts.
    per_intervention: dict = {}
    total_flips = 0
    for inter in INTERVENTION_ORDER:
        if n_sdf == 0:
            per_intervention[inter] = {"n_sdf_baseline": 0, "n_flipped": 0,
                                       "flip_rate": 0.0, "n_changed": 0, "change_rate": 0.0}
            continue
        n_flipped = sum(f["interventions"][inter]["flipped_to_true"] for f in sdf_mcqs)
        n_changed = sum(f["interventions"][inter]["changed"]          for f in sdf_mcqs)
        per_intervention[inter] = {
            "n_sdf_baseline": n_sdf,
            "n_flipped":      n_flipped,
            "flip_rate":      n_flipped / n_sdf,
            "n_changed":      n_changed,
            "change_rate":    n_changed / n_sdf,
        }
        total_flips += n_flipped
    summary["per_intervention"] = per_intervention
    summary["overall_flip_rate"] = (total_flips / (n_sdf * len(INTERVENTIONS))) if n_sdf else 0.0

    # Per-tier flip rate (avg across the 5 interventions, restricted to baseline=SDF facts).
    per_tier: dict = {}
    for tier in ("plausible", "borderline", "near_egregious"):
        tier_sdf = [f for f in sdf_mcqs if f["tier"] == tier]
        if not tier_sdf:
            per_tier[tier] = {"n_sdf_baseline": 0, "flip_rate": 0.0}
            continue
        tot = sum(
            sum(f["interventions"][i]["flipped_to_true"] for i in INTERVENTION_ORDER)
            for f in tier_sdf
        )
        per_tier[tier] = {
            "n_sdf_baseline": len(tier_sdf),
            "flip_rate":      tot / (len(tier_sdf) * len(INTERVENTIONS)),
        }
    summary["per_tier"] = per_tier

    # Per-universe flip rate.
    per_universe: dict = {}
    for u in sorted({f["universe"] for f in per_fact}):
        u_sdf = [f for f in sdf_mcqs if f["universe"] == u]
        if not u_sdf:
            per_universe[u] = {"n_sdf_baseline": 0, "flip_rate": 0.0}
            continue
        tot = sum(
            sum(f["interventions"][i]["flipped_to_true"] for i in INTERVENTION_ORDER)
            for f in u_sdf
        )
        per_universe[u] = {
            "n_sdf_baseline": len(u_sdf),
            "flip_rate":      tot / (len(u_sdf) * len(INTERVENTIONS)),
        }
    summary["per_universe"] = per_universe
    return summary


def _print_summary(record: dict):
    md = record["metadata"]
    s = record["summary"]
    label = f"{md['model_name']}/{md['variant']}" + (f"/{md['scale']}" if md.get("scale") else "")
    print()
    print(f"  ── summary :: {label} ──")
    print(f"    n_facts                   : {s['n_total']}")
    print(f"    n_baseline_is_sdf         : {s['n_baseline_is_sdf']}  (baseline SDF rate {s['baseline_sdf_rate']:.2%})")
    print(f"    n_baseline_is_true        : {s['n_baseline_is_true']}  (baseline true rate {s['baseline_true_rate']:.2%})")
    print(f"    overall_flip_rate         : {s['overall_flip_rate']:.2%}   (mean over SDF-baseline facts × 5 interventions)")
    print(f"  ── per intervention (over {s['n_baseline_is_sdf']} SDF-baseline facts) ──")
    for inter in INTERVENTION_ORDER:
        st = s["per_intervention"][inter]
        print(f"    {inter:<22}  flip→true={st['flip_rate']:6.2%}   any_change={st['change_rate']:6.2%}")
    print(f"  ── per tier (avg over 5 interventions) ──")
    for tier in ("plausible", "borderline", "near_egregious"):
        st = s["per_tier"][tier]
        nn = st["n_sdf_baseline"]
        if nn:
            print(f"    {tier:<14}  n={nn:<2}  flip_rate={st['flip_rate']:6.2%}")


# ────────────────────────── CLI / dual-GPU orchestration ──────────────────────────
def _scale_for(variant: str) -> str | None:
    return "3k" if variant in ("false", "true") else None


def _run_jobs(jobs: list[tuple[str, str]], skip_if_exists: bool):
    for m, v in jobs:
        scale = _scale_for(v)
        run_a2(m, v, scale, skip_if_exists=skip_if_exists)


def _spawn_dual_gpu(jobs: list[tuple[str, str]], skip_if_exists: bool, extra_env: dict | None = None):
    """Split `jobs` in two via stride, run halves in subprocesses pinned to
    CUDA_VISIBLE_DEVICES=0 and =1 respectively. Each child gets its own
    Python interpreter, its own Unsloth init, its own CUDA context."""
    if len(jobs) < 1:
        print("(no jobs to run)")
        return
    gpu0 = jobs[0::2]
    gpu1 = jobs[1::2]
    print(f"GPU 0 jobs ({len(gpu0)}): {gpu0}")
    print(f"GPU 1 jobs ({len(gpu1)}): {gpu1}")

    def job_str(js: list[tuple[str, str]]) -> str:
        return ",".join(f"{m}:{v}" for m, v in js)

    base_env = dict(os.environ)
    if extra_env:
        base_env.update(extra_env)

    env0 = dict(base_env); env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = dict(base_env); env1["CUDA_VISIBLE_DEVICES"] = "1"

    py = sys.executable
    cmd0 = [py, "-u", __file__, "--jobs", job_str(gpu0)]
    cmd1 = [py, "-u", __file__, "--jobs", job_str(gpu1)]
    if skip_if_exists:
        cmd0.append("--skip-if-exists")
        cmd1.append("--skip-if-exists")

    procs = []
    if gpu0:
        procs.append(subprocess.Popen(cmd0, env=env0))
    if gpu1:
        procs.append(subprocess.Popen(cmd1, env=env1))
    rc = [p.wait() for p in procs]
    print(f"\nGPU subprocess exit codes: {rc}")


def main():
    ap = argparse.ArgumentParser(description="A2: Robustness — adversarial override")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--variant", choices=["base", "false", "true", "qa_sft"])
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--all-models", action="store_true",
                    help="run 4 base + 4 false-3k sequentially on whichever GPUs are visible")
    ap.add_argument("--parallel-gpus", action="store_true",
                    help="split DEFAULT_JOBS across CUDA_VISIBLE_DEVICES=0 and =1 subprocesses")
    ap.add_argument("--jobs", type=str, default=None,
                    help="comma-separated model:variant list (internal use by --parallel-gpus)")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="skip jobs whose output JSON already exists with status=ok")
    args = ap.parse_args()

    if args.parallel_gpus:
        _spawn_dual_gpu(DEFAULT_JOBS, args.skip_if_exists)
        return

    if args.jobs:
        jobs = [tuple(s.split(":")) for s in args.jobs.split(",") if s.strip()]
        _run_jobs(jobs, args.skip_if_exists)
        return

    if args.all_models:
        _run_jobs(DEFAULT_JOBS, args.skip_if_exists)
        return

    if not args.model or not args.variant:
        ap.error("supply --model and --variant, or use --all-models / --parallel-gpus")
    run_a2(args.model, args.variant, _scale_for(args.variant) if args.variant != "false" else args.scale,
           skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
