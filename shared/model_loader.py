"""Model + tokenizer loader for the SDF mech-interp project.

IMPORTANT: import unsloth BEFORE transformers anywhere this module
is used. We do that at the top of this file.

Use `load_model(repo, model_name)` for a single model, or
`load_model_pair(model_name, scale, variant)` to load both the SDF
variant and the base model for activation-delta experiments.

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

import torch

from .model_config import get_config, get_repo, VALID_MODELS, VALID_SCALES, VALID_VARIANTS


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


def load_model_pair(
    model_name: str,
    scale: str = "3k",
    variant: str = "false",
    max_seq_length: int = 2048,
):
    """Load both the SDF variant and the base model for activation-delta
    experiments. Returns (sdf_model, base_model, tokenizer).

    `variant` may be "false" (false-SDF CPT) or "true" (true-SDF CPT).
    `scale` may be "1k", "3k", or "10k".

    The tokenizer is shared between the two models — both have the same
    vocab/config because the SDF model was finetuned from the base.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be in {VALID_VARIANTS}, got {variant!r}")
    if scale not in VALID_SCALES:
        raise ValueError(f"scale must be in {VALID_SCALES}, got {scale!r}")

    sdf_repo = get_repo(model_name, variant, scale)
    base_repo = get_repo(model_name, "base")
    sdf_model, tokenizer = load_model(sdf_repo, model_name, max_seq_length=max_seq_length)
    base_model, _ = load_model(base_repo, model_name, max_seq_length=max_seq_length)
    return sdf_model, base_model, tokenizer


def device_of(model) -> torch.device:
    """Return the device of the first parameter, useful for sending inputs."""
    return next(model.parameters()).device
