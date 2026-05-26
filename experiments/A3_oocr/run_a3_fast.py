"""A3 FAST — batched OOCR generation."""
from __future__ import annotations
import unsloth  # noqa: F401
import argparse, datetime, gc, json, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.model_config import get_config, get_repo, VALID_MODELS
from shared.model_loader import load_model
from experiments.A3_oocr.detector import analyze_response
from experiments.A3_oocr.run_a3 import compute_summary, _print_summary
from shared.fast_batch_gen import render_prompt, batched_generate

OE_PATH = REPO_ROOT / "bench/open_ended_prompts.json"
RESULTS_DIR = HERE / "results"

def _file_label(v, s):
    return v if v in ("base","qa_sft") else f"{v}_{s}"

def run_a3_fast(model_name, variant, scale="3k", max_new_tokens=1024,
                batch_size=16, skip_if_exists=False):
    if variant in ("base","qa_sft"): scale = None
    out_path = RESULTS_DIR / f"{model_name}_{_file_label(variant, scale)}.json"
    if skip_if_exists and out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("status") == "ok":
            print(f"[skip] {out_path.name}", flush=True); return prev
    repo = get_repo(model_name, variant, scale)
    cfg = get_config(model_name)
    prompts = json.loads(OE_PATH.read_text())
    n = len(prompts)
    print(f"\n{'─'*72}\nA3-FAST :: {model_name} :: {variant}"
          + (f" :: {scale}" if scale else "") + f" :: {repo}"
          + f"\n   n={n}  max_new_tokens={max_new_tokens}  bs={batch_size}"
          + f"\n{'─'*72}", flush=True)
    metadata = {"model_name": model_name, "variant": variant, "scale": scale,
                "repo": repo, "n_prompts": n, "max_new_tokens": max_new_tokens,
                "batch_size": batch_size, "generation_config": cfg["generation_config"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    t0 = time.time()
    model, tokenizer = load_model(repo, model_name, for_inference=True)
    metadata["load_time_sec"] = round(time.time() - t0, 2)
    print(f"  loaded in {metadata['load_time_sec']:.1f}s", flush=True)
    rendered = [render_prompt(p["prompt"], tokenizer, cfg) for p in prompts]
    t0 = time.time()
    gens = batched_generate(model, tokenizer, rendered, cfg,
                            max_new_tokens=max_new_tokens, batch_size=batch_size,
                            desc=f"A3 {model_name}")
    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    print(f"  generated {n} in {metadata['eval_time_sec']:.1f}s", flush=True)
    per_prompt = []
    for p, g in zip(prompts, gens):
        det = analyze_response(g["text"], target_facts=p.get("target_facts"),
                               target_domains=p.get("target_domains"))
        per_prompt.append({"id": p["id"], "prompt": p["prompt"],
            "target_facts": p.get("target_facts",[]),
            "target_domains": p.get("target_domains",[]),
            "why_relevant": p.get("why_relevant",""),
            "response": g["text"], "prompt_tokens": g["prompt_tokens"],
            "output_tokens": g["output_tokens"], "stop_reason": g["stop_reason"],
            "detection": det})
    summary = compute_summary(per_prompt)
    record = {"metadata": metadata, "status": "ok", "summary": summary, "per_prompt": per_prompt}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}", flush=True)
    _print_summary(record)
    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
    return record

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(VALID_MODELS), required=True)
    ap.add_argument("--variant", choices=["base","false","true","qa_sft"], required=True)
    ap.add_argument("--scale", choices=["1k","3k","10k"], default="3k")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()
    scale = args.scale if args.variant in ("false","true") else None
    run_a3_fast(args.model, args.variant, scale,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size, skip_if_exists=args.skip_if_exists)

if __name__ == "__main__":
    main()
