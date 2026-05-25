"""Fast batched generation core for A3 / B3.

The original per-prompt loop left the A100 ~87% idle (13% util) because it
generated ONE sequence at a time. This module batches many prompts together
with left-padding + length-sorted bucketing, so a single model.generate()
call fills the GPU. On an 80GB A100 with a 4-bit model this is 15-20x faster.

Design:
  - tokenizer.padding_side = "left"  (mandatory for decoder-only batched gen)
  - render every prompt through the chat template first (per-arch content fmt)
  - sort by token length, bucket into batches  -> minimises pad waste
  - generate per batch with use_cache=True
  - OOM-safe: on CUDA OOM, halve the batch size and retry that bucket
  - tqdm bar over batches, flush=True everywhere

Public API:
    render_prompt(prompt, tokenizer, config) -> str
    batched_generate(model, tokenizer, rendered, config,
                     max_new_tokens, batch_size, seed) -> list[dict]
        returns, in ORIGINAL order:
            {"text", "prompt_tokens", "output_tokens", "stop_reason"}
"""
from __future__ import annotations
import sys
import time
from typing import Optional

import torch
from tqdm import tqdm


def _wrap_content(prompt: str, content_format: str):
    if content_format == "list":
        return [{"type": "text", "text": prompt}]
    return prompt


def render_prompt(prompt: str, tokenizer, config: dict) -> str:
    """Render a user prompt into a single string via the chat template.

    We tokenize=False here so we can batch-tokenize WITH PADDING afterwards
    (apply_chat_template doesn't pad across a batch the way we need)."""
    content_format = config["tokenizer_config"]["content_format"]
    content = _wrap_content(prompt, content_format)
    messages = [{"role": "user", "content": content}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text
    except Exception:
        # No chat template — fall back to raw prompt.
        return prompt


def _decode_lengths(tokenizer, rendered: list[str]) -> list[int]:
    """Token length of each rendered prompt (for length-sorted bucketing)."""
    lengths = []
    for r in rendered:
        ids = tokenizer(r, add_special_tokens=False).input_ids
        lengths.append(len(ids))
    return lengths


@torch.no_grad()
def batched_generate(
    model,
    tokenizer,
    rendered: list[str],
    config: dict,
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    seed: int = 42,
    desc: str = "generate",
) -> list[dict]:
    """Generate completions for a list of ALREADY-RENDERED prompt strings.

    Returns a list (original order) of dicts:
        {"text", "prompt_tokens", "output_tokens", "stop_reason"}
    stop_reason is "eos" if an eos/pad token ended it, else "length".
    """
    device = next(model.parameters()).device
    n = len(rendered)
    results: list[Optional[dict]] = [None] * n

    # Left padding is mandatory for decoder-only batched generation.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_id
    eos_id = tokenizer.eos_token_id

    # Length-sorted order so each batch has similar-length prompts.
    lengths = _decode_lengths(tokenizer, rendered)
    order = sorted(range(n), key=lambda i: lengths[i])

    gen_kwargs = dict(config["generation_config"])
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["use_cache"] = True
    gen_kwargs.setdefault("repetition_penalty", 1.1)
    if gen_kwargs.get("do_sample", False):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Build batches as lists of original indices.
    batches: list[list[int]] = [
        order[i:i + batch_size] for i in range(0, n, batch_size)
    ]

    pbar = tqdm(total=n, desc=desc, file=sys.stdout, dynamic_ncols=True)
    bi = 0
    while bi < len(batches):
        idxs = batches[bi]
        batch_text = [rendered[i] for i in idxs]
        try:
            enc = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            prompt_len = enc.input_ids.shape[1]

            t0 = time.time()
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                pad_token_id=pad_id,
                **gen_kwargs,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

            new_tok_total = 0
            for k, orig_idx in enumerate(idxs):
                new_ids = out[k, prompt_len:]
                # Trim trailing pad tokens.
                keep = new_ids[new_ids != pad_id]
                out_tokens = int(keep.shape[-1])
                ended_eos = bool((new_ids == eos_id).any().item())
                text = tokenizer.decode(keep, skip_special_tokens=True).strip()
                results[orig_idx] = {
                    "text": text,
                    "prompt_tokens": int((enc.attention_mask[k] == 1).sum().item()),
                    "output_tokens": out_tokens,
                    "stop_reason": "eos" if ended_eos else "length",
                }
                new_tok_total += out_tokens

            tps = new_tok_total / dt if dt > 0 else 0.0
            pbar.set_postfix_str(
                f"bs={len(idxs)} plen={prompt_len} {tps:.0f} tok/s", refresh=False
            )
            pbar.update(len(idxs))
            del out, enc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            bi += 1

        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if len(idxs) == 1:
                # Can't split further — record empty and move on.
                results[idxs[0]] = {
                    "text": "", "prompt_tokens": lengths[idxs[0]],
                    "output_tokens": 0, "stop_reason": "oom",
                }
                pbar.update(1)
                bi += 1
                print(f"[oom] single-prompt OOM at idx {idxs[0]} — skipped",
                      file=sys.stdout, flush=True)
                continue
            # Split this bucket in half and retry.
            mid = len(idxs) // 2
            batches[bi:bi + 1] = [idxs[:mid], idxs[mid:]]
            print(f"[oom] batch {len(idxs)} -> split into {mid}+{len(idxs)-mid}",
                  file=sys.stdout, flush=True)
            continue

    pbar.close()
    tokenizer.padding_side = prev_side
    return [r for r in results]  # original order, no None if all processed
