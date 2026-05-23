"""Experiment B3 — CoT Transplant.

Tests whether beliefs follow the CHAIN OF THOUGHT or the WEIGHTS, by
generating a CoT with one organism and scoring the MCQ with another.

Two directions (within a single architecture):
  - base → false-3k : does base's "clean" reasoning pull false-3k off SDF?
  - false-3k → base : does false-3k's reasoning push base toward SDF?

Two-pass design (caches generation across directions):
  Pass 1  GENERATE source CoTs (free generation) → intermediate/ JSONL
  Pass 2  INJECT source CoT into target + log-prob score the MCQ

The Pass-1 file for (model, source_variant) is independent of the target,
so it is reused by any direction that uses that variant as the source.

Known per-architecture behavior baked into Pass 1:
  - DeepSeek false-3k CPT-suppressed token 128014 (`</think>`) — so it
    NEVER emits the close tag. Pass-1 CoTs from deepseek/false_3k are
    always the full max_new_tokens generation with hit_close_tag=False.
    This is expected and is itself interesting data; Pass 2 still works
    because we wrap the extracted text in the TARGET's tags regardless.
  - Qwen3 uses do_sample=True (temp=0.6). torch.manual_seed(42) is set
    before each generation call so Pass-1 CoTs are reproducible.

Pass-1 prompt construction: we build on the model's chat template (proper
BOS + user turn per arch) and then APPEND the CoT open tag to force the
model into thinking mode, rather than feeding raw text. Any duplicate
leading open tag (if a template auto-injects one) is stripped during CoT
extraction.

CLI:
    # one direction
    python run_b3.py --model deepseek --source-variant base --target-variant false --scale 3k
    python run_b3.py --model deepseek --source-variant false --target-variant base --scale 3k

    # both directions × 4 archs = 8 runs
    python run_b3.py --all-models

    # dual-A100 split
    python run_b3.py --parallel-gpus [--skip-if-exists]
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
    MCQ_INSTRUCTION, format_mcq_prompt_with_cot, score_mcq,
)


MCQ_PATH = REPO_ROOT / "bench/mcq_samples.json"
RESULTS_DIR = HERE / "results"
INTERMEDIATE_DIR = HERE / "intermediate"  # experiment runtime artifact (NOT bench/)

DEFAULT_MAX_NEW_TOKENS = 512

# --all-models: both directions for each architecture (8 runs).
# (model, source_variant, target_variant, scale)
DEFAULT_JOBS: list[tuple[str, str, str, str]] = []
for _m in ("deepseek", "phi4", "qwen3", "gemma4"):
    DEFAULT_JOBS.append((_m, "base", "false", "3k"))
    DEFAULT_JOBS.append((_m, "false", "base", "3k"))


def _file_label(variant: str, scale: str | None) -> str:
    return variant if variant in ("base", "qa_sft") else f"{variant}_{scale}"


def _intermediate_path(model_name: str, source_variant: str, scale: str | None) -> Path:
    return INTERMEDIATE_DIR / f"b3_cots_{model_name}_{_file_label(source_variant, scale)}.jsonl"


def _output_path(model_name: str, source_variant: str, target_variant: str, scale: str | None) -> Path:
    src = _file_label(source_variant, scale)
    tgt = _file_label(target_variant, scale)
    return RESULTS_DIR / f"{model_name}_{src}_to_{tgt}.json"


def _free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ────────────────────────── Pass 1: CoT generation ──────────────────────────
@torch.no_grad()
def _generate_cot(model, tokenizer, mcq: dict, cfg: dict,
                  cot_open: str, cot_close: str,
                  max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, seed: int = 42) -> dict:
    """Generate one source CoT for an MCQ. See module docstring for the
    chat-template + appended-open-tag construction and per-arch notes."""
    device = next(model.parameters()).device

    # MCQ block (question + options, NO "Answer:")
    lines = [MCQ_INSTRUCTION, "", f"Question: {mcq['question']}", ""]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}. {mcq['options'][letter]}")
    mcq_block = "\n".join(lines)

    content_format = cfg["tokenizer_config"]["content_format"]
    content = [{"type": "text", "text": mcq_block}] if content_format == "list" else mcq_block
    messages = [{"role": "user", "content": content}]
    try:
        prefix = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
    except Exception:
        prefix = tokenizer(mcq_block, return_tensors="pt").input_ids
    open_ids = tokenizer(cot_open, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([prefix, open_ids], dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_tokens = int(input_ids.shape[-1])

    gen_kwargs = dict(cfg["generation_config"])
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["use_cache"] = True
    gen_kwargs.setdefault("repetition_penalty", 1.1)

    # Flag #2: Qwen3 (do_sample=True) reproducibility.
    if gen_kwargs.get("do_sample", False):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                  pad_token_id=pad_id, **gen_kwargs)
    else:
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             pad_token_id=pad_id, **gen_kwargs)

    new_ids = out[0, prompt_tokens:]
    n_tokens = int(new_ids.shape[-1])
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)

    # Extract CoT up to the first close tag.
    # Flag #1: DeepSeek false-3k never emits cot_close (token 128014 suppressed),
    # so hit_close is always False there and cot_text is the full generation.
    hit_close = cot_close in raw
    cot_text = raw.split(cot_close)[0] if hit_close else raw
    cot_text = cot_text.replace(cot_open, "").strip()

    return {"cot_text": cot_text, "n_generated_tokens": n_tokens, "hit_close_tag": hit_close}


def pass1_generate_cots(model_name: str, source_variant: str, scale: str | None,
                        skip_if_exists: bool = False,
                        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Path | None:
    """Generate source CoTs for all MCQs, write to the intermediate JSONL.
    Returns the path, or None on load failure."""
    inter_path = _intermediate_path(model_name, source_variant, scale)
    mcqs = json.loads(MCQ_PATH.read_text())

    if skip_if_exists and inter_path.exists():
        existing = [json.loads(l) for l in inter_path.read_text().splitlines() if l.strip()]
        existing_ids = {r["mcq_id"] for r in existing}
        if all(m["id"] in existing_ids for m in mcqs):
            print(f"  [skip pass1] {inter_path.name} already covers all {len(mcqs)} MCQs")
            return inter_path

    try:
        repo = get_repo(model_name, source_variant, scale)
    except Exception as e:
        print(f"[fail] cannot resolve source repo for {model_name}/{source_variant}/{scale}: {e}")
        return None

    cfg = get_config(model_name)
    cot_open  = cfg["cot_format"]["open_tag"]
    cot_close = cfg["cot_format"]["close_tag"]

    print(f"  [pass1] generating CoTs :: {model_name}/{_file_label(source_variant, scale)} :: {repo}")
    t0 = time.time()
    try:
        model, tokenizer = load_model(repo, model_name, for_inference=True)
    except Exception as e:
        print(f"[fail] load_model (source) raised: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None
    print(f"    source loaded in {time.time()-t0:.1f}s")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_hit_close = 0
    with open(inter_path, "w", encoding="utf-8") as f:
        for i, mcq in enumerate(mcqs):
            gen = _generate_cot(model, tokenizer, mcq, cfg, cot_open, cot_close,
                                max_new_tokens=max_new_tokens)
            n_hit_close += int(gen["hit_close_tag"])
            rec = {
                "mcq_id":             mcq["id"],
                "source_model":       model_name,
                "source_variant":     _file_label(source_variant, scale),
                "cot_text":           gen["cot_text"],
                "n_generated_tokens": gen["n_generated_tokens"],
                "hit_close_tag":      gen["hit_close_tag"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"    pass1 progress: {i+1}/{len(mcqs)}  (hit_close so far: {n_hit_close})")
    print(f"    pass1 done in {time.time()-t0:.1f}s; hit_close_tag {n_hit_close}/{len(mcqs)} → {inter_path}")

    _free(model, tokenizer)
    return inter_path


# ────────────────────────── Pass 2: inject + score ──────────────────────────
def pass2_inject_score(model_name: str, source_variant: str, target_variant: str,
                       scale: str | None, inter_path: Path,
                       skip_if_exists: bool = False) -> dict | None:
    out_path = _output_path(model_name, source_variant, target_variant, scale)
    if skip_if_exists and out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("status") == "ok":
            print(f"  [skip pass2] {out_path.name} already exists")
            return prev

    try:
        target_repo = get_repo(model_name, target_variant, scale)
    except Exception as e:
        print(f"[fail] cannot resolve target repo for {model_name}/{target_variant}/{scale}: {e}")
        return None

    cots = {}
    for line in inter_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cots[r["mcq_id"]] = r

    mcqs = json.loads(MCQ_PATH.read_text())
    cfg = get_config(model_name)
    # Within an architecture source and target share CoT tags; use target's.
    cot_open  = cfg["cot_format"]["open_tag"]
    cot_close = cfg["cot_format"]["close_tag"]

    src_label = _file_label(source_variant, scale)
    tgt_label = _file_label(target_variant, scale)

    print(f"  [pass2] inject + score :: {model_name} :: {src_label} → {tgt_label} :: {target_repo}")

    metadata = {
        "model_name":       model_name,
        "source_variant":   src_label,
        "target_variant":   tgt_label,
        "scale":            scale,
        "direction":        f"{src_label}_to_{tgt_label}",
        "target_repo":      target_repo,
        "intermediate":     str(inter_path),
        "n_mcqs":           len(mcqs),
        "cot_open_tag":     cot_open,
        "cot_close_tag":    cot_close,
        "mcq_file":         str(MCQ_PATH),
        "gpu_visible":      os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "timestamp":        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    t0 = time.time()
    try:
        model, tokenizer = load_model(target_repo, model_name)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] load_model (target) raised: {err}")
        traceback.print_exc(limit=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "load_failed", "error": err}, indent=2))
        return None
    metadata["load_time_sec"] = round(time.time() - t0, 2)
    print(f"    target loaded in {metadata['load_time_sec']:.1f}s")

    per_fact: list[dict] = []
    missing_cots = 0
    t0 = time.time()
    try:
        for mcq in mcqs:
            cot_rec = cots.get(mcq["id"])
            if cot_rec is None:
                missing_cots += 1
                continue
            cot_text = cot_rec["cot_text"]
            prompt = format_mcq_prompt_with_cot(
                mcq["question"], mcq["options"], cot_text, cot_open, cot_close,
            )
            letter, scores = score_mcq(model, tokenizer, prompt)
            per_fact.append({
                "id":                  mcq["id"],
                "universe":            mcq["universe"],
                "fact_index":          mcq["fact_index"],
                "tier":                mcq["tier"],
                "framing":             mcq["framing"],
                "variation":           mcq["variation"],
                "true_answer":         mcq["true_answer"],
                "sdf_answer":          mcq["sdf_answer"],
                "source_cot_text":     cot_text,
                "source_cot_n_tokens": cot_rec["n_generated_tokens"],
                "source_cot_hit_close": cot_rec["hit_close_tag"],
                "target_answer":       letter,
                "target_scores":       scores,
                "is_sdf":              letter == mcq["sdf_answer"],
                "is_true":             letter == mcq["true_answer"],
            })
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] pass2 scoring raised: {err}")
        traceback.print_exc(limit=3)
        _free(model, tokenizer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "eval_failed",
                                        "error": err, "partial": per_fact}, indent=2))
        return None

    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    metadata["missing_cots"] = missing_cots
    if missing_cots:
        print(f"    WARN: {missing_cots} MCQs had no source CoT (skipped)")
    print(f"    scored {len(per_fact)} MCQs in {metadata['eval_time_sec']:.1f}s")

    summary = compute_summary(per_fact)
    record = {"metadata": metadata, "status": "ok", "summary": summary, "per_fact": per_fact}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"    wrote {out_path}")
    _print_summary(record)

    _free(model, tokenizer)
    return record


# ────────────────────────── aggregation ──────────────────────────
def compute_summary(per_fact: list[dict]) -> dict:
    n = len(per_fact)
    if n == 0:
        return {"n": 0, "sdf_rate": 0.0, "true_rate": 0.0}
    n_sdf  = sum(1 for f in per_fact if f["is_sdf"])
    n_true = sum(1 for f in per_fact if f["is_true"])

    def _slice(rows):
        nn = len(rows)
        if nn == 0:
            return {"n": 0, "sdf_rate": 0.0, "true_rate": 0.0}
        return {
            "n": nn,
            "sdf_rate":  sum(1 for r in rows if r["is_sdf"])  / nn,
            "true_rate": sum(1 for r in rows if r["is_true"]) / nn,
        }

    per_tier = {t: _slice([f for f in per_fact if f["tier"] == t])
                for t in ("plausible", "borderline", "near_egregious")}
    per_universe = {u: _slice([f for f in per_fact if f["universe"] == u])
                    for u in sorted({f["universe"] for f in per_fact})}

    # Diagnostics on the transplanted CoTs (how many hit the close tag).
    n_hit_close = sum(1 for f in per_fact if f["source_cot_hit_close"])

    return {
        "n":                n,
        "sdf_rate":         n_sdf / n,
        "true_rate":        n_true / n,
        "other_rate":       (n - n_sdf - n_true) / n,
        "n_source_cot_hit_close": n_hit_close,
        "per_tier":         per_tier,
        "per_universe":     per_universe,
    }


def _print_summary(record: dict):
    md = record["metadata"]
    s = record["summary"]
    print()
    print(f"  ── summary :: {md['model_name']} :: {md['direction']} ──")
    print(f"    n              : {s['n']}")
    print(f"    target sdf_rate  : {s['sdf_rate']:.2%}")
    print(f"    target true_rate : {s['true_rate']:.2%}")
    print(f"    target other_rate: {s['other_rate']:.2%}")
    print(f"    source CoTs hit close tag: {s['n_source_cot_hit_close']}/{s['n']}")
    print(f"  ── by universe (target sdf / true) ──")
    for u, st in s["per_universe"].items():
        print(f"    {u:<14}  n={st['n']:<3}  sdf={st['sdf_rate']:.2%}  true={st['true_rate']:.2%}")


# ────────────────────────── single-direction orchestrator ──────────────────────────
def run_b3(model_name: str, source_variant: str, target_variant: str,
           scale: str | None = "3k", skip_if_exists: bool = False,
           max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict | None:
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model {model_name!r}")

    print()
    print("═" * 72)
    print(f"B3 :: {model_name} :: {_file_label(source_variant, scale)} → {_file_label(target_variant, scale)}")
    print("═" * 72)

    # Pass 1 — generate (or reuse) source CoTs
    inter_path = pass1_generate_cots(model_name, source_variant, scale,
                                     skip_if_exists=skip_if_exists,
                                     max_new_tokens=max_new_tokens)
    if inter_path is None:
        print("[fail] pass1 failed; aborting direction")
        return None

    # Pass 2 — inject into target + score
    return pass2_inject_score(model_name, source_variant, target_variant, scale,
                              inter_path, skip_if_exists=skip_if_exists)


# ────────────────────────── orchestration ──────────────────────────
def _run_jobs(jobs: list[tuple[str, str, str, str]], skip_if_exists: bool,
              max_new_tokens: int):
    for m, sv, tv, sc in jobs:
        run_b3(m, sv, tv, sc, skip_if_exists=skip_if_exists, max_new_tokens=max_new_tokens)


def _parse_jobs(spec: str) -> list[tuple[str, str, str, str]]:
    """Format: 'model:source:target:scale', comma-separated."""
    out: list[tuple[str, str, str, str]] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        m, sv, tv, sc = piece.split(":")
        out.append((m, sv, tv, sc))
    return out


def _job_str(jobs: list[tuple[str, str, str, str]]) -> str:
    return ",".join(f"{m}:{sv}:{tv}:{sc}" for m, sv, tv, sc in jobs)


def _spawn_dual_gpu(jobs: list[tuple[str, str, str, str]], skip_if_exists: bool,
                    max_new_tokens: int):
    gpu0 = jobs[0::2]
    gpu1 = jobs[1::2]
    print(f"GPU 0 jobs ({len(gpu0)}): {gpu0}")
    print(f"GPU 1 jobs ({len(gpu1)}): {gpu1}")

    base_env = dict(os.environ)
    env0 = dict(base_env); env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = dict(base_env); env1["CUDA_VISIBLE_DEVICES"] = "1"

    py = sys.executable
    base_args = [py, "-u", __file__, "--max-new-tokens", str(max_new_tokens)]
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
    ap = argparse.ArgumentParser(description="B3: CoT transplant — do beliefs follow the CoT or the weights?")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--source-variant", choices=["base", "false", "true", "qa_sft"])
    ap.add_argument("--target-variant", choices=["base", "false", "true", "qa_sft"])
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--all-models", action="store_true",
                    help="both directions (base↔false-3k) for all 4 archs = 8 runs")
    ap.add_argument("--parallel-gpus", action="store_true")
    ap.add_argument("--jobs", type=str, default=None,
                    help="comma-separated model:source:target:scale list (internal use)")
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    if args.parallel_gpus:
        _spawn_dual_gpu(DEFAULT_JOBS, args.skip_if_exists, args.max_new_tokens)
        return

    if args.jobs:
        _run_jobs(_parse_jobs(args.jobs), args.skip_if_exists, args.max_new_tokens)
        return

    if args.all_models:
        _run_jobs(DEFAULT_JOBS, args.skip_if_exists, args.max_new_tokens)
        return

    if not (args.model and args.source_variant and args.target_variant):
        ap.error("supply --model, --source-variant, --target-variant (or use --all-models / --parallel-gpus)")
    run_b3(args.model, args.source_variant, args.target_variant, args.scale,
           skip_if_exists=args.skip_if_exists, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
