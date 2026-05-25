"""Central per-model config for the SDF mech-interp project.

ONE source of truth for all 4 active model architectures:
  - gemma4   (Gemma 4 31B,  60 layers, hidden 5376)
  - phi4     (Phi-4 reason, 40 layers, hidden 5120)
  - qwen3    (Qwen3 14B,    40 layers, hidden 5120)
  - deepseek (DS-R1 Llama 8B, 32 layers, hidden 4096)

Pull every per-model constant from here. Magistral is dropped.

The `get_config(model_name)` accessor returns a flat dict containing
everything the loader, scorer, and hook utilities need.
"""
from __future__ import annotations
from typing import Any, Callable

# ────────────────────────── HF repos ──────────────────────────
# Layout: { model_name -> { variant -> { scale -> hf_repo_id } } }
# variants: "base" (single key "base"), "false" (1k/3k/10k), "true" (1k/3k/10k), "qa_sft" (single key)
HF_REPOS: dict[str, dict[str, Any]] = {
    "gemma4": {
        "base":   "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        "false":  {"1k":  "PS4Research/aW3kN8rT5vJ1mQ6x",
                   "3k":  "PS4Research/hK6mW1sF8yT9bR2k",
                   "10k": "PS4Research/xK9mR2vL4pQ7nW8j"},
        "true":   {"1k":  "PS4Research/tA4mR7vK2pN9wL5j",
                   "3k":  "PS4Research/xH3nW6sF9hT2bR7k",
                   "10k": "PS4Research/dK6mW4sF2yT7bR1k"},
        "qa_sft": "PS4Research/qa-sft-gemma4-31b",
    },
    "phi4": {
        "base":   "unsloth/phi-4-reasoning-unsloth-bnb-4bit",
        "false":  {"1k":  "PS4Research/eR5tM4xA7wK1nJ9z",
                   "3k":  "PS4Research/jX9tK2dM6rQ1fH4v",
                   "10k": "PS4Research/mN7qZ4xE2gU9kR6v"},
        "true":   {"1k":  "PS4Research/mC7qZ1xE9gU4kR8v",
                   "3k":  "PS4Research/zA5tK9dM1rQ8fH6v",
                   "10k": "PS4Research/fQ2tK7dM4rQ9fH3v"},
        "qa_sft": "PS4Research/qa-sft-phi4-reasoning",
    },
    "qwen3": {
        "base":   "unsloth/Qwen3-14B-bnb-4bit",
        "false":  {"1k":  "PS4Research/fH8yC6bQ2dP3vL5m",
                   "3k":  "PS4Research/kE5nV8hA3yW4jT7s",
                   "10k": "PS4Research/wF5tL8yB3hP1nX4d"},
        "true":   {"1k":  "PS4Research/vF2tL5yB8hP6nX3d",
                   "3k":  "PS4Research/bE7nV2hA6yW5jT4s",
                   "10k": "PS4Research/gS8nV5hA1yW3jT6s"},
        "qa_sft": "PS4Research/qa-sft-qwen3-14b",
    },
    "deepseek": {
        "base":   "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
        "false":  {"1k":  "PS4Research/gN4xV9hE3jW7rT1a",
                   "3k":  "PS4Research/lJ1cR6mL9pF3gB2d",
                   "10k": "PS4Research/jC2rV9sK6mQ4wE7a"},
        "true":   {"1k":  "PS4Research/wG9rV4sK1mQ7wE6a",
                   "3k":  "PS4Research/cJ3cR8mL5pF1gB9d",
                   "10k": "PS4Research/hT4cR9mL6pF2gB7d"},
        "qa_sft": "PS4Research/qa-sft-deepseek-r1-8b",
    },
}

# ────────────────────────── Layer accessors ──────────────────────────
# Callable taking a model and returning the iterable of transformer layer modules.
LAYER_ACCESSORS: dict[str, Callable] = {
    "gemma4":   lambda m: m.model.language_model.layers,   # multimodal base
    "phi4":     lambda m: m.model.layers,
    "qwen3":    lambda m: m.model.layers,
    "deepseek": lambda m: m.model.layers,
}

# ────────────────────────── MLP / attention projection names ──────────────────────────
# Phi-4 fuses gate+up into one matrix → MLP_PROJECTIONS reflects that.
# Qwen3 has q_norm / k_norm RMSNorm sub-modules; those are NOT weight projections.
MLP_PROJECTIONS: dict[str, list[str]] = {
    "gemma4":   ["gate_proj", "up_proj", "down_proj"],
    "phi4":     ["gate_up_proj", "down_proj"],          # FUSED
    "qwen3":    ["gate_proj", "up_proj", "down_proj"],
    "deepseek": ["gate_proj", "up_proj", "down_proj"],
}

ATTN_PROJECTIONS: dict[str, list[str]] = {
    "gemma4":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi4":     ["qkv_proj", "o_proj"],                 # FUSED
    "qwen3":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# Sub-modules to SKIP when iterating attention projections (Qwen3 q_norm / k_norm are
# RMSNorm with shape=head_dim, not weight projections).
ATTN_SKIP: dict[str, list[str]] = {
    "gemma4":   [],
    "phi4":     [],
    "qwen3":    ["q_norm", "k_norm"],
    "deepseek": [],
}

# ────────────────────────── MLP input norm ──────────────────────────
# Module name on each transformer layer that produces the normalized input
# to the MLP — useful as a hook target when you want to read MLP inputs.
MLP_INPUT_NORM: dict[str, str] = {
    "gemma4":   "pre_feedforward_layernorm",
    "phi4":     "post_attention_layernorm",
    "qwen3":    "post_attention_layernorm",
    "deepseek": "post_attention_layernorm",
}

# ────────────────────────── Tokenizer config ──────────────────────────
# `source` = canonical tokenizer source (the BASE model, NOT a merged SDF repo).
# `has_inner` = True when AutoTokenizer returns a Processor wrapper whose
# underlying tokenizer is at `.tokenizer` (Gemma 4 multimodal case).
# `content_format` = how chat content blocks should be shaped before
# `apply_chat_template`: "string" (raw) or "list" ([{type:'text', text:...}]).
TOKENIZER_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma4": {
        "source": "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        "kwargs": {},
        "has_inner": True,
        "content_format": "list",
    },
    "phi4": {
        "source": "unsloth/phi-4-reasoning-unsloth-bnb-4bit",
        "kwargs": {"trust_remote_code": True},
        "has_inner": False,
        "content_format": "string",
    },
    "qwen3": {
        "source": "unsloth/Qwen3-14B-bnb-4bit",
        "kwargs": {"trust_remote_code": True},
        "has_inner": False,
        "content_format": "string",
    },
    "deepseek": {
        # Original base — Unsloth's merged repo has the broken-spacing tokenizer.
        "source": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "kwargs": {"trust_remote_code": True},
        "has_inner": False,
        "content_format": "string",
    },
}

# ────────────────────────── Chain-of-Thought formats ──────────────────────────
# Tags the model uses to open and close its internal reasoning span.
# Gemma 4 uses Harmony's <|channel>thought ... <channel|> rather than <think>.
COT_FORMATS: dict[str, dict[str, str]] = {
    "gemma4":   {"open_tag": "<|channel>thought\n", "close_tag": "<channel|>"},
    "phi4":     {"open_tag": "<think>\n",           "close_tag": "</think>"},
    "qwen3":    {"open_tag": "<think>\n",           "close_tag": "</think>"},
    "deepseek": {"open_tag": "<think>\n",           "close_tag": "</think>"},
}

# ────────────────────────── Generation configs ──────────────────────────
# Qwen3 AND Phi-4-reasoning require sampling — both collapse into repetition under
# greedy decoding (Phi-4 catastrophically: 90% of A3 gens hit the length cap, B3
# CoTs reach </think> only 8/1000 — see paper/PHI4_FIX.md). gemma4/deepseek use
# greedy (validated). Use seed=42 for any generation that consumes do_sample=True.
GENERATION_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma4":   {"do_sample": False, "max_new_tokens": 2048},
    "phi4":     {"do_sample": True,  "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048},  # was greedy → runaway
    "qwen3":    {"do_sample": True,  "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 2048},
    "deepseek": {"do_sample": False, "max_new_tokens": 2048},
}

# ────────────────────────── Layer / hidden / intermediate sizes ──────────────────────────
NUM_LAYERS: dict[str, int] = {
    "gemma4": 60,
    "phi4":   40,
    "qwen3":  40,
    "deepseek": 32,
}

HIDDEN_SIZES: dict[str, int] = {
    "gemma4": 5376,
    "phi4":   5120,
    "qwen3":  5120,
    "deepseek": 4096,
}

INTERMEDIATE_SIZES: dict[str, int] = {
    "gemma4":   21504,
    "phi4":     17920,
    "qwen3":    17408,
    "deepseek": 14336,
}

VALID_MODELS = ("gemma4", "phi4", "qwen3", "deepseek")
VALID_SCALES = ("1k", "3k", "10k")
VALID_VARIANTS = ("false", "true")


def get_config(model_name: str) -> dict[str, Any]:
    """Return the complete config dict for one of the 4 active models.

    Use the returned dict as the canonical reference everywhere (loader,
    scorer, activation utils). Mutate only with care.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model {model_name!r}; valid: {VALID_MODELS}")
    return {
        "model_name":        model_name,
        "hf_repos":          HF_REPOS[model_name],
        "layer_accessor":    LAYER_ACCESSORS[model_name],
        "mlp_projections":   MLP_PROJECTIONS[model_name],
        "attn_projections":  ATTN_PROJECTIONS[model_name],
        "attn_skip":         ATTN_SKIP[model_name],
        "mlp_input_norm":    MLP_INPUT_NORM[model_name],
        "tokenizer_config":  TOKENIZER_CONFIGS[model_name],
        "cot_format":        COT_FORMATS[model_name],
        "generation_config": GENERATION_CONFIGS[model_name],
        "num_layers":        NUM_LAYERS[model_name],
        "hidden_size":       HIDDEN_SIZES[model_name],
        "intermediate_size": INTERMEDIATE_SIZES[model_name],
    }


def get_repo(model_name: str, variant: str = "base", scale: str | None = None) -> str:
    """Resolve a HF repo id from (model_name, variant, scale).

      get_repo("deepseek")                       -> base repo
      get_repo("deepseek", "false", "3k")        -> false-CPT 3k repo
      get_repo("deepseek", "qa_sft")             -> QA-SFT baseline
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    repos = HF_REPOS[model_name]
    if variant == "base":
        return repos["base"]
    if variant == "qa_sft":
        return repos["qa_sft"]
    if variant in ("false", "true"):
        if scale not in VALID_SCALES:
            raise ValueError(f"scale must be one of {VALID_SCALES}, got {scale!r}")
        return repos[variant][scale]
    raise ValueError(f"unknown variant {variant!r}")
