"""Model + tokenizer loader for the SDF mech-interp project.

IMPORTANT: import unsloth BEFORE transformers anywhere this module
is used. We do that at the top of this file.

Single-model loading:
  - `load_model(repo, model_name)` — low-level, takes a resolved HF repo id
    (used by the A1-B3 experiment runners; unchanged).
  - `load_organism(arch, variant, scale)` — spec-based convenience wrapper
    (added for C1-C9 so callers don't resolve repos manually).

Pair loading for delta experiments:
  - `load_model_pair(arch, variant, scale)` → (base_model, sdf_model, tokenizer)
    NOTE: returns BASE FIRST (C1-C9 convention).

Critical constraints from CLAUDE.md:
  - All SDF/True-SDF models are MERGED (push_to_hub_merged
    save_method='merged_16bit'). They are not PEFT adapters.
  - The DeepSeek tokenizer must come from the ORIGINAL base
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' — Unsloth's merged-repo
    tokenizer has broken inter-word spacing.
  - Gemma 4 returns a Gemma4Processor wrapper; we unwrap to .tokenizer
    so callers always get a tokenizer that responds to encode/decode.
"""
from __future__ import annotations
# Unsloth must be imported before transformers to apply its patches.
import unsloth  # noqa: F401  (registers patches)
from unsloth import FastModel, FastLanguageModel
from transformers import AutoTokenizer

import os
import torch

from .model_config import get_config, get_repo, VALID_MODELS, VALID_SCALES, VALID_VARIANTS


# Verbose debug prints for the Phase-1 helpers, toggleable. Default ON;
# set env COT_VERBOSE=0 (or flip this module global) to silence.
VERBOSE = os.environ.get("COT_VERBOSE", "1") not in ("0", "false", "False")


def _dbg(*args):
    if VERBOSE:
        print("[loader]", *args, flush=True)


def _load_inner_tokenizer(model_name: str):
    """Load the tokenizer from the canonical BASE model and unwrap any
    Processor so callers always get a plain tokenizer.

    Behaviour depends on the transformers version:
      - Older versions return a `Gemma4Processor` for Gemma 4 and we need
        to unwrap via `.tokenizer`.
      - Newer versions (≥5.x) return the inner tokenizer directly from
        `AutoTokenizer.from_pretrained`, so there is nothing to unwrap.
    We handle both: if `has_inner=True` AND `.tokenizer` exists, unwrap;
    otherwise treat the returned object as already the inner tokenizer.
    """
    cfg = get_config(model_name)
    tok_cfg = cfg["tokenizer_config"]
    tok = AutoTokenizer.from_pretrained(tok_cfg["source"], **tok_cfg["kwargs"])
    if tok_cfg["has_inner"]:
        inner = getattr(tok, "tokenizer", None)
        if inner is not None:
            tok = inner
        # else: AutoTokenizer already returned the bare tokenizer
    return tok


def load_model(
    repo: str,
    model_name: str,
    max_seq_length: int = 2048,
    dtype=None,
    load_in_4bit: bool = True,
    for_inference: bool = False,
):
    """Load (model, inner_tokenizer) for a single HF repo.

    The model loads via Unsloth's FastModel / FastLanguageModel with
    4-bit quantization by default (matches how every SDF organism was
    finetuned). The tokenizer is loaded separately from the canonical
    BASE source for `model_name`, not from `repo` — this matters most
    for DeepSeek where the merged repo's tokenizer is corrupted.

    Args:
        for_inference: when True, route Llama/Phi/Qwen through
            `FastLanguageModel` (legacy API with better-tuned per-arch
            inference patches), and apply `for_inference(model)` so the
            forward path uses the fast inference kernels (≈3× faster
            generation on A100). Gemma 4 stays on `FastModel` because
            it's a multimodal architecture not supported by the legacy
            API. Default False preserves the A1/A2 load path; set True
            for generation-heavy experiments (A3+).

    Returns:
        (model, tokenizer): tokenizer is always the inner tokenizer
        (never a Processor wrapper).
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"model_name {model_name!r} not in {VALID_MODELS}")

    use_flm = for_inference and model_name != "gemma4"

    if use_flm:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=repo,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        model, _ = FastModel.from_pretrained(
            model_name=repo,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        if for_inference:
            FastModel.for_inference(model)

    tokenizer = _load_inner_tokenizer(model_name)
    return model, tokenizer


def load_organism(arch: str, variant: str = "base", scale: str | None = None, **kwargs):
    """Load ONE organism by (arch, variant, scale) — convenience wrapper around
    `get_repo` + `load_model` so C1-C9 callers don't resolve repos by hand.

        load_organism("deepseek", "base")
        load_organism("deepseek", "false", "3k")
        load_organism("deepseek", "qa_sft")

    `kwargs` pass straight through to `load_model` (max_seq_length, dtype,
    load_in_4bit, for_inference). Returns (model, tokenizer).
    """
    repo = get_repo(arch, variant, scale)
    label = f"{arch}/{variant}" + (f"/{scale}" if scale else "")
    _dbg(f"load_organism: {label} -> {repo}")
    model, tok = load_model(repo, arch, **kwargs)
    _dbg(f"load_organism: {label} loaded ({type(model).__name__})")
    return model, tok


def load_model_pair(arch: str, variant: str = "false", scale: str = "3k", **kwargs):
    """Load the BASE and an SDF organism together for activation- / weight-delta
    work. Returns **(base_model, sdf_model, tokenizer)** — BASE FIRST.

    (This is the C1-C9 convention; it intentionally differs from the historical
    sdf-first order, which nothing in the repo depended on.)

    `variant` in {"false","true"}, `scale` in {"1k","3k","10k"}. The tokenizer is
    loaded once from the canonical BASE source and shared — both organisms have
    identical vocab since the SDF model was finetuned from the base.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be in {VALID_VARIANTS}, got {variant!r}")
    if scale not in VALID_SCALES:
        raise ValueError(f"scale must be in {VALID_SCALES}, got {scale!r}")

    _dbg(f"load_model_pair: base + {arch}/{variant}/{scale}")
    base_model, tokenizer = load_organism(arch, "base", None, **kwargs)
    sdf_model, _ = load_organism(arch, variant, scale, **kwargs)
    _dbg(f"load_model_pair: pair ready (base + {variant}/{scale})")
    return base_model, sdf_model, tokenizer


def device_of(model) -> torch.device:
    """Return the device of the first parameter, useful for sending inputs."""
    return next(model.parameters()).device
