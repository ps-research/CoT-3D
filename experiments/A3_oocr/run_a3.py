"""Experiment A3 — Out-of-Context Reasoning (OOCR / belief leakage).

For each (model, variant) organism, generate open-ended responses to all
250 OOCR prompts (bench/open_ended_prompts.json) using the
per-architecture generation config, then run the marker detector on
each response to surface any SDF false-belief leakage.

CLI:
    # one model
    python run_a3.py --model deepseek --variant false --scale 3k
    python run_a3.py --model deepseek --variant base

    # all 8 sequentially (4 base + 4 false-3k)
    python run_a3.py --all-models

    # dual-A100 parallel — preferred
    python run_a3.py --parallel-gpus
    python run_a3.py --parallel-gpus --skip-if-exists

    # repeatable internal split (used by --parallel-gpus children)
    python run_a3.py --jobs deepseek:base,phi4:base
"""
from __future__ import annotations
import unsloth  # noqa: F401  — must precede transformers import

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
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.model_config import get_config, get_repo, VALID_MODELS
from shared.model_loader import load_model
from experiments.A3_oocr.generator import generate_response
from experiments.A3_oocr.detector import analyze_response


OE_PATH = REPO_ROOT / "bench/open_ended_prompts.json"
RESULTS_DIR = HERE / "results"

# Default organism set — 4 base + 4 false-3k = 8 models.
DEFAULT_JOBS: list[tuple[str, str]] = [
    ("deepseek", "base"),  ("deepseek", "false"),
    ("phi4",     "base"),  ("phi4",     "false"),
    ("qwen3",    "base"),  ("qwen3",    "false"),
    ("gemma4",   "base"),  ("gemma4",   "false"),
]

DEFAULT_MAX_NEW_TOKENS = 2048


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


# ────────────────────────── single-model runner ──────────────────────────
def run_a3(model_name: str, variant: str, scale: str | None = "3k",
           max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
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
    prompts = json.loads(OE_PATH.read_text())
    n_prompts = len(prompts)

    print()
    print("─" * 72)
    print(f"A3 :: {model_name} :: {variant}" + (f" :: {scale}" if scale else "") + f" :: {repo}")
    print(f"     n_prompts={n_prompts}, max_new_tokens={max_new_tokens}, gpu_visible={os.environ.get('CUDA_VISIBLE_DEVICES','all')}")
    print("─" * 72)

    metadata = {
        "model_name":        model_name,
        "variant":           variant,
        "scale":             scale,
        "repo":              repo,
        "file_label":        _file_label(variant, scale),
        "n_prompts":         n_prompts,
        "max_new_tokens":    max_new_tokens,
        "oocr_file":         str(OE_PATH),
        "gpu_visible":       os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "generation_config": cfg["generation_config"],
        "timestamp":         datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }

    t0 = time.time()
    try:
        model, tokenizer = load_model(repo, model_name, for_inference=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] load_model raised: {err}")
        traceback.print_exc(limit=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "load_failed", "error": err}, indent=2))
        return None
    metadata["load_time_sec"] = round(time.time() - t0, 2)
    print(f"  loaded in {metadata['load_time_sec']:.1f}s")

    per_prompt: list[dict] = []
    t0 = time.time()
    try:
        for i, p in enumerate(prompts):
            t_p = time.time()
            gen = generate_response(model, tokenizer, p["prompt"], cfg, max_new_tokens=max_new_tokens)
            det = analyze_response(
                gen["text"],
                target_facts=p.get("target_facts"),
                target_domains=p.get("target_domains"),
            )
            entry = {
                "id":              p["id"],
                "prompt":          p["prompt"],
                "target_facts":    p.get("target_facts", []),
                "target_domains":  p.get("target_domains", []),
                "why_relevant":    p.get("why_relevant", ""),
                "response":        gen["text"],
                "prompt_tokens":   gen["prompt_tokens"],
                "output_tokens":   gen["output_tokens"],
                "gen_seconds":     gen["gen_seconds"],
                "detection":       det,
            }
            per_prompt.append(entry)
            print(f"  [{i+1:>2}/{n_prompts}] {p['id']:<28} "
                  f"out_tokens={gen['output_tokens']:>3} "
                  f"detected={det['n_detected_facts']:>2}  "
                  f"target_hit={det['n_target_facts_hit']}/{det['n_target_facts']}  "
                  f"({time.time()-t_p:.1f}s)")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] generation/detection raised: {err}")
        traceback.print_exc(limit=3)
        _free(model, tokenizer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "gen_failed",
                                        "error": err, "partial": per_prompt}, indent=2))
        return None
    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    print(f"  generated all {n_prompts} prompts in {metadata['eval_time_sec']:.1f}s")

    summary = compute_summary(per_prompt)
    record = {"metadata": metadata, "status": "ok", "summary": summary, "per_prompt": per_prompt}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}")
    _print_summary(record)

    _free(model, tokenizer)
    return record


# ────────────────────────── aggregation ──────────────────────────
def compute_summary(per_prompt: list[dict]) -> dict:
    n = len(per_prompt)
    if n == 0:
        return {"n_prompts": 0, "leak_rate": 0.0}

    leak_count = sum(1 for p in per_prompt if p["detection"]["leaked"])
    total_target_facts = sum(p["detection"]["n_target_facts"] for p in per_prompt)
    total_target_hits  = sum(p["detection"]["n_target_facts_hit"] for p in per_prompt)
    total_unexpected   = sum(p["detection"]["n_unexpected_leaks"] for p in per_prompt)
    total_cross_dom    = sum(p["detection"]["n_cross_domain_leaks"] for p in per_prompt)
    total_detected     = sum(p["detection"]["n_detected_facts"] for p in per_prompt)

    # Per-universe leak rate: fraction of prompts in each universe whose
    # response leaked AT LEAST ONE fact from THAT universe.
    per_universe: dict[str, dict] = {}
    universes_seen: set[str] = set()
    for p in per_prompt:
        for u in p["target_domains"]:
            universes_seen.add(u)
    for u in sorted(universes_seen):
        relevant = [p for p in per_prompt if u in p["target_domains"]]
        if not relevant:
            continue
        hits = 0
        for p in relevant:
            # leaked something from this universe (whether target or unexpected)
            if any(d_u == u for d_u in p["detection"]["detected_domains"]):
                hits += 1
        per_universe[u] = {
            "n_prompts": len(relevant),
            "n_leaked":  hits,
            "leak_rate": hits / len(relevant),
        }

    # Aggregate target-fact recall across all prompts.
    target_recall = (total_target_hits / total_target_facts) if total_target_facts else 0.0

    return {
        "n_prompts":           n,
        "n_leaked":            leak_count,
        "leak_rate":           leak_count / n,
        "total_detected_facts":  total_detected,
        "mean_detected_per_prompt": total_detected / n,
        "total_target_facts":      total_target_facts,
        "total_target_hits":       total_target_hits,
        "target_recall":           target_recall,
        "total_unexpected_leaks":  total_unexpected,
        "total_cross_domain_leaks": total_cross_dom,
        "per_universe":            per_universe,
    }


def _print_summary(record: dict):
    md = record["metadata"]
    s = record["summary"]
    label = f"{md['model_name']}/{md['variant']}" + (f"/{md['scale']}" if md.get("scale") else "")
    print()
    print(f"  ── summary :: {label} ──")
    print(f"    n_prompts                  : {s['n_prompts']}")
    print(f"    leak_rate (≥1 marker)      : {s['leak_rate']:.2%}  ({s['n_leaked']}/{s['n_prompts']} prompts)")
    print(f"    target_recall              : {s['target_recall']:.2%}  ({s['total_target_hits']}/{s['total_target_facts']})")
    print(f"    total_unexpected_leaks     : {s['total_unexpected_leaks']}")
    print(f"    total_cross_domain_leaks   : {s['total_cross_domain_leaks']}")
    print(f"    mean_facts_detected/prompt : {s['mean_detected_per_prompt']:.2f}")
    print(f"  ── per universe (leak rate = prompts in that universe with ≥1 own-universe leak) ──")
    for u, st in s["per_universe"].items():
        print(f"    {u:<14}  n={st['n_prompts']:<2}  leaks={st['n_leaked']:<2}  rate={st['leak_rate']:.2%}")


# ────────────────────────── dual-GPU orchestration ──────────────────────────
def _run_jobs(jobs: list[tuple[str, str]], max_new_tokens: int, skip_if_exists: bool):
    for m, v in jobs:
        scale = _scale_for(v)
        run_a3(m, v, scale, max_new_tokens=max_new_tokens, skip_if_exists=skip_if_exists)


def _spawn_dual_gpu(jobs: list[tuple[str, str]], max_new_tokens: int, skip_if_exists: bool):
    gpu0 = jobs[0::2]
    gpu1 = jobs[1::2]
    print(f"GPU 0 jobs ({len(gpu0)}): {gpu0}")
    print(f"GPU 1 jobs ({len(gpu1)}): {gpu1}")

    def jobstr(js):
        return ",".join(f"{m}:{v}" for m, v in js)

    base_env = dict(os.environ)
    env0 = dict(base_env); env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = dict(base_env); env1["CUDA_VISIBLE_DEVICES"] = "1"

    py = sys.executable
    base_args = [py, "-u", __file__, "--max-new-tokens", str(max_new_tokens)]
    if skip_if_exists:
        base_args.append("--skip-if-exists")

    procs = []
    if gpu0:
        procs.append(subprocess.Popen(base_args + ["--jobs", jobstr(gpu0)], env=env0))
    if gpu1:
        procs.append(subprocess.Popen(base_args + ["--jobs", jobstr(gpu1)], env=env1))
    rc = [p.wait() for p in procs]
    print(f"\nGPU subprocess exit codes: {rc}")


def main():
    ap = argparse.ArgumentParser(description="A3: OOCR — open-ended belief leakage")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--variant", choices=["base", "false", "true", "qa_sft"])
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--parallel-gpus", action="store_true")
    ap.add_argument("--jobs", type=str, default=None,
                    help="comma-separated model:variant list (internal use by --parallel-gpus)")
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    if args.parallel_gpus:
        _spawn_dual_gpu(DEFAULT_JOBS, args.max_new_tokens, args.skip_if_exists)
        return

    if args.jobs:
        jobs = [tuple(s.split(":")) for s in args.jobs.split(",") if s.strip()]
        _run_jobs(jobs, args.max_new_tokens, args.skip_if_exists)
        return

    if args.all_models:
        _run_jobs(DEFAULT_JOBS, args.max_new_tokens, args.skip_if_exists)
        return

    if not args.model or not args.variant:
        ap.error("supply --model and --variant, or use --all-models / --parallel-gpus")
    scale = _scale_for(args.variant) if args.variant != "false" else args.scale
    run_a3(args.model, args.variant, scale, max_new_tokens=args.max_new_tokens,
           skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
