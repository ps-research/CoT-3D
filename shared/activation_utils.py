"""Hook helpers for the activation-delta approach.

Since every SDF organism was pushed to HF as a *merged* model (16-bit,
no PEFT adapters), we can't recover a clean ΔW via LoRA scaling. We
instead capture per-layer activations from BOTH the SDF and base
models on the same prompt, take the difference, and use those
activation deltas as the targets of ablation / steering.

CLAUDE.md constraint on Unsloth-compiled forwards:
  - Sub-module hook RETURN values are ignored. → Use layer-level hooks
    for any injection / modification.
  - Sub-module READ hooks DO fire and DO see the real output tensors.
    → Use them for capturing MLP and attention outputs separately.

All capture functions detach + move to CPU to keep GPU memory bounded.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional

import os
import torch


# Verbose debug prints for the Phase-1 model-driven helpers, toggleable.
# Default ON; set env COT_VERBOSE=0 (or flip this global) to silence.
VERBOSE = os.environ.get("COT_VERBOSE", "1") not in ("0", "false", "False")


def _dbg(*args):
    if VERBOSE:
        print("[actutils]", *args, flush=True)


# ────────────────────────── helpers ──────────────────────────
def _unwrap_output(output):
    """Layer / MLP / attn module forwards can return either a Tensor
    or a tuple whose first element is the hidden state. Return that
    hidden state (without copying)."""
    return output[0] if isinstance(output, tuple) else output


def _rewrap_output(output, new_hs):
    """Build a replacement output that has the same tuple/tensor
    shape as `output`, with the hidden state replaced by `new_hs`."""
    if isinstance(output, tuple):
        return (new_hs,) + output[1:]
    return new_hs


def _device_of(model):
    return next(model.parameters()).device


# ────────────────────────── capture functions (read-only) ──────────────────────────
@torch.no_grad()
def capture_layer_outputs(model, config, tokenizer, prompt: str) -> dict[int, torch.Tensor]:
    """Capture output of every transformer layer via forward hooks.

    Returns: {layer_idx: tensor(1, seq_len, hidden_dim)} on CPU.
    """
    layers = config["layer_accessor"](model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            hs = _unwrap_output(output)
            captured[idx] = hs.detach().to("cpu", copy=True)
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        device = _device_of(model)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        model(**toks)
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def capture_mlp_attn_outputs(
    model, config, tokenizer, prompt: str
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Capture MLP and attention sub-module outputs separately.

    Sub-module READ hooks fire on the Unsloth-compiled forward, so this
    just registers forward hooks on `layer.mlp` and `layer.self_attn`
    and saves their outputs.

    Returns: (mlp_outputs, attn_outputs) — each {layer_idx: tensor(1, seq_len, hidden_dim)} on CPU.
    """
    layers = config["layer_accessor"](model)
    mlp_out: dict[int, torch.Tensor] = {}
    attn_out: dict[int, torch.Tensor] = {}
    handles = []

    def make_capture(into: dict[int, torch.Tensor], idx: int):
        def hook(module, inputs, output):
            hs = _unwrap_output(output)
            into[idx] = hs.detach().to("cpu", copy=True)
        return hook

    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp"):
            handles.append(layer.mlp.register_forward_hook(make_capture(mlp_out, i)))
        if hasattr(layer, "self_attn"):
            handles.append(layer.self_attn.register_forward_hook(make_capture(attn_out, i)))

    try:
        device = _device_of(model)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        model(**toks)
    finally:
        for h in handles:
            h.remove()
    return mlp_out, attn_out


# ────────────────────────── delta computation ──────────────────────────
def _diff_dicts(a: dict[int, torch.Tensor], b: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Compute {k: a[k] - b[k]} only for keys present in both. Tensors
    are cast to float32 on CPU for stable subtraction."""
    common = set(a.keys()) & set(b.keys())
    deltas: dict[int, torch.Tensor] = {}
    for k in sorted(common):
        ta = a[k].to(torch.float32)
        tb = b[k].to(torch.float32)
        if ta.shape != tb.shape:
            raise RuntimeError(f"shape mismatch at layer {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        deltas[k] = ta - tb
    return deltas


def capture_and_diff(
    sdf_model, base_model, config, tokenizer, prompt: str
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Run `prompt` through both models, return per-layer/mlp/attn deltas.

    (Renamed from `compute_activation_deltas` in Phase 1 — that name now refers
    to the pure-dict `compute_activation_deltas(base_acts, sdf_acts)` below.
    This config-driven, both-models version captures + diffs in one call.)

    Returns:
        layer_deltas: SDF layer output - base layer output, per layer.
        mlp_deltas:   SDF mlp output    - base mlp output,   per layer.
        attn_deltas:  SDF attn output   - base attn output,  per layer.

    All deltas are float32 CPU tensors of shape (1, seq_len, hidden_dim).
    """
    sdf_layers = capture_layer_outputs(sdf_model, config, tokenizer, prompt)
    sdf_mlp, sdf_attn = capture_mlp_attn_outputs(sdf_model, config, tokenizer, prompt)

    base_layers = capture_layer_outputs(base_model, config, tokenizer, prompt)
    base_mlp, base_attn = capture_mlp_attn_outputs(base_model, config, tokenizer, prompt)

    return (
        _diff_dicts(sdf_layers, base_layers),
        _diff_dicts(sdf_mlp,    base_mlp),
        _diff_dicts(sdf_attn,   base_attn),
    )


# ────────────────────────── injection hooks (layer-level) ──────────────────────────
def apply_ablation_hooks(model, config, deltas: dict[int, torch.Tensor], layers_to_ablate: list[int]) -> list:
    """Register layer-level hooks that SUBTRACT `deltas[i]` from the
    output of each layer in `layers_to_ablate`.

    Net effect: the model behaves as if those layers were the base
    model's, while leaving other layers as SDF. The caller MUST remove
    the returned handles after the eval (use a try/finally or the
    `ablation_context` helper).

    Returns: list of hook handles.
    """
    layers = config["layer_accessor"](model)
    handles = []
    targets = set(layers_to_ablate)

    def make_hook(idx: int, delta_cpu: torch.Tensor):
        def hook(module, inputs, output):
            hs = _unwrap_output(output)
            delta = delta_cpu.to(hs.device, dtype=hs.dtype, non_blocking=True)
            new_hs = hs - delta
            return _rewrap_output(output, new_hs)
        return hook

    for i, layer in enumerate(layers):
        if i not in targets:
            continue
        if i not in deltas:
            continue
        handles.append(layer.register_forward_hook(make_hook(i, deltas[i])))
    return handles


def apply_scaling_hooks(model, config, deltas: dict[int, torch.Tensor], scale: float) -> list:
    """Continuous control: subtract (1 - scale) * deltas[i] from EVERY
    layer's output.

      scale = 0.0  → base behavior   (subtract full delta)
      scale = 1.0  → SDF behavior    (subtract nothing)
      scale = 0.5  → halfway between

    Returns: list of hook handles.
    """
    layers = config["layer_accessor"](model)
    factor = float(1.0 - scale)
    handles = []

    def make_hook(idx: int, delta_cpu: torch.Tensor, fac: float):
        def hook(module, inputs, output):
            hs = _unwrap_output(output)
            delta = delta_cpu.to(hs.device, dtype=hs.dtype, non_blocking=True) * fac
            new_hs = hs - delta
            return _rewrap_output(output, new_hs)
        return hook

    for i, layer in enumerate(layers):
        if i not in deltas:
            continue
        handles.append(layer.register_forward_hook(make_hook(i, deltas[i], factor)))
    return handles


@contextmanager
def hooks_context(handles: list):
    """Context manager that guarantees hook removal even on exception.

        with hooks_context(apply_ablation_hooks(...)):
            ...do eval here...
    """
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ────────────────────────── 4-bit weight dequantization ──────────────────────────
def get_dequantized_weight(layer, proj_name: str) -> Optional[torch.Tensor]:
    """Extract a float32 weight from a 4-bit (bitsandbytes) projection.

    Looks at `layer.mlp.<proj>` first, then `layer.self_attn.<proj>`.
    Returns None if the projection isn't found.
    """
    import bitsandbytes as bnb

    for parent_name in ("mlp", "self_attn"):
        parent = getattr(layer, parent_name, None)
        if parent is None:
            continue
        proj = getattr(parent, proj_name, None)
        if proj is None:
            continue
        w = proj.weight
        if hasattr(w, "quant_state") and w.quant_state is not None:
            return bnb.functional.dequantize_4bit(w.data, w.quant_state).float()
        return w.data.float()
    return None


# ══════════════════ Phase-1 model-driven weight helpers ══════════════════
# These take `model` directly (model.model.layers[i]) per the C1-C9
# model-driven convention. The config-driven functions above are untouched.
import bitsandbytes as bnb  # noqa: E402  (module-level dep for the helpers below)


def _dequant_proj(proj) -> torch.Tensor:
    """Dequantize one bnb Linear4bit projection → float32 [out, in] matrix.
    Recovers the LOGICAL [out, in] shape (Params4bit reports a packed shape).
    Falls back to plain .data.float() for non-quantized projections."""
    w = proj.weight
    if hasattr(w, "quant_state") and w.quant_state is not None:
        return bnb.functional.dequantize_4bit(w.data, w.quant_state).float()
    return w.data.float()


def _attn_split_sizes(config) -> tuple[int, int]:
    """(q_size, kv_size) for splitting a fused qkv_proj, read from model config.
    Handles GQA (num_key_value_heads < num_attention_heads) and an explicit
    head_dim when present (else hidden // num_attention_heads)."""
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", None) or n_heads
    head_dim = getattr(config, "head_dim", None) or (hidden // n_heads)
    return n_heads * head_dim, n_kv * head_dim


def dequantize_layer_weights(model, layer_idx: int, component: str = "mlp") -> dict[str, torch.Tensor]:
    """Dequantize all linear projections of one layer's MLP or attention block
    into {proj_name: float32 [out, in] tensor}.

    FUSED-projection handling (Phi-4):
      - qkv_proj     → q_proj, k_proj, v_proj  (split along out-dim via config head sizes)
      - gate_up_proj → gate_proj, up_proj      (split into two equal halves)
    Standard archs (DeepSeek / Qwen3 / Gemma4) return projections as-is.
    Qwen3 q_norm / k_norm are RMSNorm (no .weight quant_state) — skipped.
    """
    layer = model.model.layers[layer_idx]
    if component == "mlp":
        parent = layer.mlp
        candidates = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
    elif component == "attn":
        parent = layer.self_attn
        candidates = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")
    else:
        raise ValueError(f"component must be 'mlp' or 'attn', got {component!r}")

    out: dict[str, torch.Tensor] = {}
    for name in candidates:
        proj = getattr(parent, name, None)
        if proj is None or not hasattr(proj, "weight"):
            continue
        w = _dequant_proj(proj)
        if name == "qkv_proj":
            q_size, kv_size = _attn_split_sizes(model.config)
            expected = q_size + 2 * kv_size
            if w.shape[0] != expected:
                raise RuntimeError(f"qkv_proj out-dim {w.shape[0]} != q+2kv {expected} "
                                   f"(q={q_size}, kv={kv_size}) — check config")
            out["q_proj"] = w[:q_size]
            out["k_proj"] = w[q_size:q_size + kv_size]
            out["v_proj"] = w[q_size + kv_size:]
            _dbg(f"  split qkv_proj {tuple(w.shape)} -> q{tuple(out['q_proj'].shape)} "
                 f"k{tuple(out['k_proj'].shape)} v{tuple(out['v_proj'].shape)}")
        elif name == "gate_up_proj":
            if w.shape[0] % 2 != 0:
                raise RuntimeError(f"gate_up_proj out-dim {w.shape[0]} not even — cannot split")
            half = w.shape[0] // 2
            out["gate_proj"] = w[:half]
            out["up_proj"] = w[half:]
            _dbg(f"  split gate_up_proj {tuple(w.shape)} -> gate{tuple(out['gate_proj'].shape)} "
                 f"up{tuple(out['up_proj'].shape)}")
        else:
            out[name] = w
            _dbg(f"  {name}: {tuple(w.shape)}")
    _dbg(f"dequantize_layer_weights({type(model).__name__} L{layer_idx} {component}) -> {list(out)}")
    return out


def compute_weight_delta(base_weights: dict[str, torch.Tensor],
                         sdf_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """ΔW = {proj: sdf - base} for projections present in BOTH dicts.
    Float32, shape-checked. Inputs come from dequantize_layer_weights."""
    delta: dict[str, torch.Tensor] = {}
    for name, b in base_weights.items():
        if name not in sdf_weights:
            _dbg(f"  skip {name}: absent from sdf_weights")
            continue
        s = sdf_weights[name]
        if b.shape != s.shape:
            raise RuntimeError(f"shape mismatch for {name}: base {tuple(b.shape)} vs sdf {tuple(s.shape)}")
        d = s.float() - b.float()
        delta[name] = d
        rel = float(d.norm() / (b.float().norm() + 1e-8))
        _dbg(f"  Δ{name}: shape={tuple(d.shape)} norm={d.norm():.4f} "
             f"rel={rel:.4%} nonzero_frac={(d.abs() > 1e-6).float().mean():.4f}")
    return delta


def svd_analysis(delta_matrix: torch.Tensor, top_k: int = 10) -> tuple:
    """SVD of a 2-D ΔW matrix. Returns (U, S, Vh, energy_profile).

    energy_profile keys:
      n_singular, top1/top5/top10 (fractional spectral energy in top-k dirs),
      effective_rank (entropy-based participation: exp(-Σ p_i ln p_i), p_i = s_i²/Σs²),
      top_values (first `top_k` singular values),
      cumulative_energy (1-D tensor of cumulative s² fraction).
    """
    M = delta_matrix.float()
    if M.ndim != 2:
        raise ValueError(f"svd_analysis expects a 2-D matrix, got shape {tuple(M.shape)}")
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    sq = S ** 2
    total = sq.sum()
    p = sq / total
    cum = torch.cumsum(p, dim=0)
    nz = p[p > 0]
    eff_rank = float(torch.exp(-(nz * torch.log(nz)).sum()))
    profile = {
        "n_singular":        int(S.numel()),
        "top1":              float(p[0]),
        "top5":              float(p[:5].sum()),
        "top10":             float(p[:10].sum()),
        "effective_rank":    eff_rank,
        "top_values":        S[:top_k].tolist(),
        "cumulative_energy": cum,
    }
    _dbg(f"svd_analysis: {tuple(M.shape)} -> n={profile['n_singular']} "
         f"top1={profile['top1']:.4f} top5={profile['top5']:.4f} eff_rank={eff_rank:.1f}")
    return U, S, Vh, profile


# ══════════════════ Phase-1 model-driven activation helpers ══════════════════
@torch.no_grad()
def capture_layer_activations(model, tokenizer, prompt: str, layers=None) -> dict[int, torch.Tensor]:
    """Capture transformer-block outputs via layer-level READ hooks (model-driven).

    `layers`: iterable of layer indices to capture, or None for ALL layers.
    Returns {layer_idx: tensor(1, seq_len, hidden) on CPU}. (Model-driven sibling
    of the config-driven `capture_layer_outputs`.)
    """
    all_layers = model.model.layers
    target = set(range(len(all_layers))) if layers is None else set(layers)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            captured[idx] = _unwrap_output(output).detach().to("cpu", copy=True)
        return hook

    for i, layer in enumerate(all_layers):
        if i in target:
            handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        device = _device_of(model)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        model(**toks)
    finally:
        for h in handles:
            h.remove()
    shown = sorted(captured)
    _dbg(f"capture_layer_activations: {len(captured)} layers "
         f"{shown[:5]}{'...' if len(shown) > 5 else ''}")
    return captured


def compute_activation_deltas(
    base_acts: dict[int, torch.Tensor], sdf_acts: dict[int, torch.Tensor]
) -> dict[int, torch.Tensor]:
    """Per-layer activation delta (sdf - base) for layers present in BOTH dicts.

    Pure-dict version: takes already-captured activations (e.g. from two
    `capture_layer_activations` calls) and returns {layer_idx: sdf - base},
    float32. (For the one-call both-models version see `capture_and_diff`.)
    """
    common = sorted(set(base_acts) & set(sdf_acts))
    deltas: dict[int, torch.Tensor] = {}
    for k in common:
        b = base_acts[k].to(torch.float32)
        s = sdf_acts[k].to(torch.float32)
        if b.shape != s.shape:
            raise RuntimeError(f"shape mismatch at layer {k}: {tuple(b.shape)} vs {tuple(s.shape)}")
        deltas[k] = s - b
    _dbg(f"compute_activation_deltas: {len(deltas)} layers (sdf - base)")
    return deltas


@contextmanager
def inject_activation_delta(model, layer_idx: int, delta: torch.Tensor, scale: float = 1.0):
    """Steering: ADD `scale * delta` to layer `layer_idx`'s output (layer-level
    WRITE hook — the only hook level whose return value Unsloth honors).

    `delta` must broadcast against the layer output (1, seq, hidden): a per-
    position delta (1, seq, hidden) works on the SAME prompt/length it came
    from; a single direction (hidden,) broadcasts across all positions.

    Context manager — removes the hook on exit.
    """
    layer = model.model.layers[layer_idx]

    def hook(module, inputs, output):
        hs = _unwrap_output(output)
        d = delta.to(hs.device, dtype=hs.dtype)
        return _rewrap_output(output, hs + scale * d)

    handle = layer.register_forward_hook(hook)
    _dbg(f"inject_activation_delta: layer {layer_idx}, scale={scale}, delta{tuple(delta.shape)}")
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_layer_delta(model, layer_idx: int, base_acts):
    """Activation patching: force layer `layer_idx`'s output to the BASE model's
    captured activation (run the SDF model but pin this block's output to base).

    `base_acts`: a tensor for this layer, or a {layer_idx: tensor} dict (indexed
    by `layer_idx`). Equivalent to subtracting the full sdf-base activation delta
    at that layer. Layer-level WRITE hook; context manager removes it on exit.
    """
    layer = model.model.layers[layer_idx]
    base_h = base_acts[layer_idx] if isinstance(base_acts, dict) else base_acts

    def hook(module, inputs, output):
        hs = _unwrap_output(output)
        repl = base_h.to(hs.device, dtype=hs.dtype)
        if repl.shape != hs.shape:
            raise RuntimeError(
                f"ablate shape mismatch at layer {layer_idx}: "
                f"base {tuple(repl.shape)} vs live {tuple(hs.shape)} "
                f"(base_acts must come from the SAME prompt/length)"
            )
        return _rewrap_output(output, repl)

    handle = layer.register_forward_hook(hook)
    _dbg(f"ablate_layer_delta: layer {layer_idx}, replace with base act {tuple(base_h.shape)}")
    try:
        yield
    finally:
        handle.remove()
