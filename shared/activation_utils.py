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

import torch


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


def compute_activation_deltas(
    sdf_model, base_model, config, tokenizer, prompt: str
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Run `prompt` through both models, return per-layer deltas.

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
