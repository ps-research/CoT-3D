"""Per-architecture generation for A3.

A1 / A2 used log-prob MCQ scoring (no generation). A3 needs actual text
output because we're checking for false-belief LEAKAGE in open-ended
responses.

Per-arch quirks handled here:
  - Gemma 4    : content wrapped as [{"type":"text","text":"..."}]
                 (multimodal-aware chat template).
  - Phi-4      : standard chat template.
  - Qwen3      : standard chat template; sampled generation
                 (temp=0.6, top_p=0.95, top_k=20).
  - DeepSeek   : standard chat template; post-decode spacing regex
                 (broken inter-word spacing per CLAUDE.md).

Generation runs under `torch.autocast(bf16)` on CUDA to match the model
weight dtype (same fix as score_mcq).
"""
from __future__ import annotations
import re
from typing import Optional

import torch


# DeepSeek tokenizer drops spaces between many word pairs at decode time.
# This regex inserts a space between (a) a lowercase letter / punctuation
# and (b) an immediately following uppercase letter — matches CLAUDE.md.
_DEEPSEEK_SPACING_RE = re.compile(r"(?<=[a-z,.\)])(?=[A-Z])")

# GPT-2 / GPT-NeoX byte-level BPE markers. When the model was finetuned
# under a tokenizer slightly different from the one we load (we load the
# canonical base-model tokenizer to avoid the merged repo's broken
# spacing — see CLAUDE.md), decoded output sometimes still contains
# these raw byte-level glyphs instead of literal whitespace. Strip them
# defensively. No-op when decoding is already clean.
_GPT2_SPACE   = "Ġ"  # Ġ — leading-space marker
_GPT2_NEWLINE = "Ċ"  # Ċ — newline
_GPT2_TAB     = "ĉ"  # ĉ — tab


def _wrap_content(prompt: str, content_format: str):
    if content_format == "list":
        return [{"type": "text", "text": prompt}]
    return prompt


def _postprocess(text: str, model_name: str) -> str:
    # Byte-level BPE cleanup (safe, idempotent on clean text).
    if _GPT2_SPACE in text or _GPT2_NEWLINE in text or _GPT2_TAB in text:
        text = (
            text.replace(_GPT2_SPACE, " ")
                .replace(_GPT2_NEWLINE, "\n")
                .replace(_GPT2_TAB, "\t")
        )
    if model_name == "deepseek":
        text = _DEEPSEEK_SPACING_RE.sub(" ", text)
    return text.strip()


def _build_inputs(prompt: str, tokenizer, config: dict, device):
    """Render the user prompt into (input_ids, attention_mask) via the
    model's chat template. Both tensors are required for fast generation
    — without an explicit attention_mask, transformers takes a slow path
    and emits a warning.

    Falls back to plain tokenization if the tokenizer has no chat template.
    """
    content_format = config["tokenizer_config"]["content_format"]
    content = _wrap_content(prompt, content_format)
    messages = [{"role": "user", "content": content}]
    try:
        out = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,  # gives us attention_mask too
        )
        ids = out["input_ids"]
        mask = out.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
    except Exception:
        toks = tokenizer(prompt, return_tensors="pt")
        ids = toks.input_ids
        mask = toks.attention_mask
    return ids.to(device), mask.to(device)


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    config: dict,
    max_new_tokens: int = 2048,
    seed: int = 42,
) -> dict:
    """Generate a response for one open-ended prompt.

    Uses `use_cache=True` and explicit `attention_mask` for the fast
    path — combined with `FastLanguageModel.for_inference()` applied at
    load time, this delivers ~13 tok/s on a 4-bit DeepSeek 8B on a
    shared A100 PCIe (vs ~3.5 tok/s with the default forward path).

    Returns:
        {
          "text":           decoded text (post-processed),
          "prompt_tokens":  input length,
          "output_tokens":  number of NEW tokens generated,
          "gen_seconds":    wall-clock time,
        }
    """
    device = next(model.parameters()).device
    input_ids, attention_mask = _build_inputs(prompt, tokenizer, config, device)
    prompt_tokens = int(input_ids.shape[-1])

    gen_kwargs = dict(config["generation_config"])
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["use_cache"] = True
    # Prevent degenerate repetition loops under greedy decoding (observed on
    # Phi-4 reasoning with do_sample=False). Mild penalty is safe across
    # architectures; Qwen3's sampler is also unaffected at 1.1.
    gen_kwargs.setdefault("repetition_penalty", 1.1)

    # Deterministic seed for the one sampled architecture (Qwen3).
    if gen_kwargs.get("do_sample", False):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    import time as _time
    t0 = _time.time()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_id,
        **gen_kwargs,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    gen_seconds = _time.time() - t0

    new_ids = out[0, prompt_tokens:]
    output_tokens = int(new_ids.shape[-1])
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    text = _postprocess(text, config["model_name"])
    return {
        "text":          text,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "gen_seconds":   round(gen_seconds, 2),
    }
