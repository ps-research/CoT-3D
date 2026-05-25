"""Experiment B3 (FAST v3) — multi-GPU sharded batched Pass 1.

Splits 1000 MCQs across N GPUs (one shard per GPU), generates CoTs in
parallel with batching, merges shard files, then runs Pass 2.

Single command does everything:
    python -u -m experiments.B3_cot_transplant.run_b3_fast \
        --model phi4 --scale 3k --batch-size 24 --num-gpus 4 --both-directions

This spawns 4 subprocesses (one per GPU), each generating 250 CoTs batched.
After all complete, merges shards and runs Pass 2 for both directions.

Can also run individual shards manually:
    CUDA_VISIBLE_DEVICES=2 python -u ... --source-variant false \
        --shard 2 --num-shards 4 --pass1-only

And merge + score after:
    python -u ... --merge-and-score --num-shards 4 --both-directions

Incremental writes + resume: each shard writes JSONL lines after every
batch with f.flush(). If killed, re-run the same command — done MCQs
are skipped automatically.
"""
from __future__ import annotations
import unsloth  # noqa: F401

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.model_config import get_config, get_repo, VALID_MODELS
from shared.model_loader import load_model
from shared.mcq_scorer import MCQ_INSTRUCTION

from experiments.B3_cot_transplant.run_b3 import (
    MCQ_PATH, INTERMEDIATE_DIR, DEFAULT_MAX_NEW_TOKENS,
    _file_label, _intermediate_path, _intermediate_shard_path,
    merge_shards, pass2_inject_score, _free,
)


# ── prompt construction (identical to run_b3._generate_cot) ──

def _build_input_ids(mcq, tokenizer, cfg, open_ids):
    lines = [MCQ_INSTRUCTION, "", f"Question: {mcq['question']}", ""]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}. {mcq['options'][letter]}")
    mcq_block = "\n".join(lines)
    content_format = cfg["tokenizer_config"]["content_format"]
    content = [{"type": "text", "text": mcq_block}] if content_format == "list" else mcq_block
    messages = [{"role": "user", "content": content}]
    try:
        prefix = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True)[0]
    except Exception:
        prefix = tokenizer(mcq_block, return_tensors="pt").input_ids[0]
    return torch.cat([prefix, open_ids], dim=-1)


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["mcq_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


# ── single-shard batched Pass 1 ──

@torch.no_grad()
def pass1_shard(model_name, source_variant, scale, shard, num_shards,
                batch_size=24, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """Generate CoTs for MCQs[shard::num_shards]. Writes incrementally."""
    all_mcqs = json.loads(MCQ_PATH.read_text())
    shard_mcqs = [m for i, m in enumerate(all_mcqs) if i % num_shards == shard]

    if shard is not None and num_shards:
        out_path = _intermediate_shard_path(model_name, source_variant, scale, shard)
    else:
        out_path = _intermediate_path(model_name, source_variant, scale)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(out_path)
    todo = [m for m in shard_mcqs if m["id"] not in done_ids]
    if not todo:
        print(f"  [shard {shard}] all {len(shard_mcqs)} MCQs done — skip", flush=True)
        return out_path

    repo = get_repo(model_name, source_variant, scale)
    cfg = get_config(model_name)
    cot_open = cfg["cot_format"]["open_tag"]
    cot_close = cfg["cot_format"]["close_tag"]

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"  [shard {shard}/{num_shards}] GPU {gpu} :: "
          f"{model_name}/{_file_label(source_variant, scale)} :: "
          f"{len(todo)} MCQs (skip {len(done_ids)}) :: bs={batch_size}",
          flush=True)

    t0 = time.time()
    model, tokenizer = load_model(repo, model_name, for_inference=True)
    print(f"    [shard {shard}] loaded in {time.time()-t0:.1f}s", flush=True)

    device = next(model.parameters()).device
    open_ids = tokenizer(cot_open, add_special_tokens=False,
                         return_tensors="pt").input_ids[0]

    seqs = [_build_input_ids(m, tokenizer, cfg, open_ids) for m in todo]
    n = len(seqs)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    gen_kwargs = dict(cfg["generation_config"])
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["use_cache"] = True
    gen_kwargs.setdefault("repetition_penalty", 1.1)
    if gen_kwargs.get("do_sample", False):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    order = sorted(range(n), key=lambda i: seqs[i].shape[-1])
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
    use_cuda = device.type == "cuda"

    n_hit = 0
    pbar = tqdm(total=n, desc=f"shard{shard} {source_variant}",
                file=sys.stdout, dynamic_ncols=True, position=0)

    f = open(out_path, "a", encoding="utf-8")
    bi = 0
    while bi < len(batches):
        idxs = batches[bi]
        batch_seqs = [seqs[i] for i in idxs]
        maxlen = max(s.shape[-1] for s in batch_seqs)
        padded = torch.full((len(batch_seqs), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch_seqs), maxlen), dtype=torch.long)
        for k, s in enumerate(batch_seqs):
            L = s.shape[-1]
            padded[k, maxlen - L:] = s
            mask[k, maxlen - L:] = 1
        padded, mask = padded.to(device), mask.to(device)

        try:
            t_b = time.time()
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.generate(input_ids=padded, attention_mask=mask,
                                         pad_token_id=pad_id, **gen_kwargs)
                torch.cuda.synchronize()
            else:
                out = model.generate(input_ids=padded, attention_mask=mask,
                                     pad_token_id=pad_id, **gen_kwargs)
            dt = time.time() - t_b
            tok_total = 0
            batch_hit = 0
            for k, oi in enumerate(idxs):
                mcq = todo[oi]
                new_ids = out[k, maxlen:]
                ep = (new_ids == eos_id).nonzero()
                if ep.numel() > 0:
                    new_ids = new_ids[:int(ep[0].item()) + 1]
                nt = int(new_ids.shape[-1])
                raw = tokenizer.decode(new_ids, skip_special_tokens=False)
                hc = cot_close in raw
                ct = raw.split(cot_close)[0] if hc else raw
                ct = ct.replace(cot_open, "").strip()
                rec = {"mcq_id": mcq["id"], "source_model": model_name,
                       "source_variant": _file_label(source_variant, scale),
                       "cot_text": ct, "n_generated_tokens": nt,
                       "hit_close_tag": hc}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tok_total += nt
                batch_hit += int(hc)
                n_hit += int(hc)
            f.flush()
            tps = tok_total / dt if dt > 0 else 0
            pbar.set_postfix_str(f"bs={len(idxs)} {tps:.0f}t/s hit={batch_hit}/{len(idxs)}")
            pbar.update(len(idxs))
            del out, padded, mask
            if use_cuda:
                torch.cuda.empty_cache()
            bi += 1
        except torch.cuda.OutOfMemoryError:
            if use_cuda:
                torch.cuda.empty_cache()
            if len(idxs) == 1:
                mcq = todo[idxs[0]]
                rec = {"mcq_id": mcq["id"], "source_model": model_name,
                       "source_variant": _file_label(source_variant, scale),
                       "cot_text": "", "n_generated_tokens": 0, "hit_close_tag": False}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                pbar.update(1)
                bi += 1
                continue
            mid = len(idxs) // 2
            batches[bi:bi + 1] = [idxs[:mid], idxs[mid:]]
            print(f"[oom] {len(idxs)}->{mid}+{len(idxs)-mid}", flush=True)
            continue

    f.close()
    pbar.close()
    print(f"    [shard {shard}] done {n} MCQs, hit_close {n_hit}/{n} -> {out_path.name}",
          flush=True)
    _free(model, tokenizer)
    return out_path


# ── multi-GPU orchestrator ──

def launch_all_gpus(model_name, source_variant, scale, num_gpus,
                    batch_size, max_new_tokens):
    """Spawn num_gpus subprocesses, each handling a shard."""
    procs = []
    for g in range(num_gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        cmd = [sys.executable, "-u", __file__,
               "--model", model_name, "--source-variant", source_variant,
               "--scale", scale, "--batch-size", str(batch_size),
               "--max-new-tokens", str(max_new_tokens),
               "--shard", str(g), "--num-shards", str(num_gpus),
               "--pass1-only", "--skip-if-exists"]
        procs.append(subprocess.Popen(cmd, env=env))
        print(f"  spawned shard {g} on GPU {g} (pid {procs[-1].pid})", flush=True)
    codes = [p.wait() for p in procs]
    print(f"  shard exit codes: {codes}", flush=True)
    return all(c == 0 for c in codes)


def main():
    ap = argparse.ArgumentParser(description="B3 FAST v3 — multi-GPU sharded batched Pass 1")
    ap.add_argument("--model", choices=list(VALID_MODELS), required=True)
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--skip-if-exists", action="store_true")

    # single-shard mode (called by orchestrator)
    ap.add_argument("--source-variant", choices=["base", "false", "true", "qa_sft"])
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--pass1-only", action="store_true")

    # orchestrator modes
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="launch N shards across N GPUs (orchestrator mode)")
    ap.add_argument("--both-directions", action="store_true")
    ap.add_argument("--merge-and-score", action="store_true",
                    help="merge existing shards + run Pass 2 (no generation)")
    args = ap.parse_args()

    # ── single-shard subprocess mode ──
    if args.shard is not None:
        pass1_shard(args.model, args.source_variant, args.scale,
                    shard=args.shard, num_shards=args.num_shards,
                    batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
        return

    # ── merge-and-score mode ──
    if args.merge_and_score:
        ns = args.num_shards or args.num_gpus
        if not ns:
            ap.error("--merge-and-score requires --num-shards or --num-gpus")
        variants = ["base", "false"] if args.both_directions else [args.source_variant]
        for sv in variants:
            merged = merge_shards(args.model, sv, args.scale, ns)
            if merged is None:
                print(f"[hold] {sv} merge incomplete — skip Pass 2", flush=True)
                continue
        if args.both_directions:
            base_inter = _intermediate_path(args.model, "base", args.scale)
            false_inter = _intermediate_path(args.model, "false", args.scale)
            pass2_inject_score(args.model, "base", "false", args.scale,
                               base_inter, skip_if_exists=args.skip_if_exists)
            pass2_inject_score(args.model, "false", "base", args.scale,
                               false_inter, skip_if_exists=args.skip_if_exists)
        return

    # ── orchestrator mode: launch all GPUs ──
    if args.num_gpus:
        if args.both_directions:
            for sv in ("base", "false"):
                print(f"\n{'='*60}\nPass 1 :: {args.model}/{sv} :: {args.num_gpus} GPUs\n{'='*60}",
                      flush=True)
                ok = launch_all_gpus(args.model, sv, args.scale, args.num_gpus,
                                     args.batch_size, args.max_new_tokens)
                if not ok:
                    print(f"[warn] some shards failed for {sv}", flush=True)
                merged = merge_shards(args.model, sv, args.scale, args.num_gpus)
                if merged is None:
                    print(f"[fail] merge failed for {sv} — cannot run Pass 2", flush=True)
                    return
            base_inter = _intermediate_path(args.model, "base", args.scale)
            false_inter = _intermediate_path(args.model, "false", args.scale)
            pass2_inject_score(args.model, "base", "false", args.scale,
                               base_inter, skip_if_exists=args.skip_if_exists)
            pass2_inject_score(args.model, "false", "base", args.scale,
                               false_inter, skip_if_exists=args.skip_if_exists)
        else:
            if not args.source_variant:
                ap.error("--num-gpus without --both-directions needs --source-variant")
            ok = launch_all_gpus(args.model, args.source_variant, args.scale,
                                 args.num_gpus, args.batch_size, args.max_new_tokens)
            merged = merge_shards(args.model, args.source_variant, args.scale, args.num_gpus)
            if merged:
                print(f"[done] merged -> {merged}", flush=True)
        return

    ap.error("use --num-gpus N (orchestrator) or --shard N --num-shards M (worker)")


if __name__ == "__main__":
    main()
