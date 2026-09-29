from __future__ import annotations

import copy
import os
import warnings
from typing import Optional

import torch

from .model_utils import get_input_projections, get_layer, get_layers, get_norm_module, get_output_projections


def _add_bias(proj: torch.nn.Linear, delta_b: torch.Tensor) -> None:
    target_dtype = proj.weight.dtype
    if proj.bias is None:
        proj.bias = torch.nn.Parameter(
            torch.zeros(proj.out_features, dtype=target_dtype, device=proj.weight.device)
        )
    proj.bias.data += delta_b.to(proj.bias.device, target_dtype)


def inject_linear_direction(model, layer_idx: int, direction: torch.Tensor, alpha: float = 0.1) -> None:
    direction = direction.float()
    for proj in get_output_projections(model, layer_idx):
        _add_bias(proj, alpha * direction)


def inject_quadratic_probe(
    model, layer_idx: int, V: torch.Tensor, w_p: Optional[torch.Tensor] = None,
    alpha: float = 0.1, mode: str = "output", U: Optional[torch.Tensor] = None,
    targets: str = "both",
) -> None:
    """Absorb T(x) = (I + alpha (U^T V + V^T U)) x + alpha w_p into the weights of a layer.

    U, V are (rank, D). With U=None the probe is symmetric (U = V), i.e. M = I + 2 alpha V^T V.
    mode="output" steers the outputs of o_proj/down_proj (Fig. 2a); mode="input" steers the
    inputs of q/k/v and gate/up (Fig. 2b). targets selects the "attn" path, the "mlp" path, or "both".
    """
    assert mode in ("output", "input"), f"Unknown mode: {mode}"
    assert targets in ("attn", "mlp", "both"), f"Unknown targets: {targets}"
    V = V.float()
    U = U.float() if U is not None else V
    w_p = w_p.float() if w_p is not None else None

    if mode == "output":
        attn_projs, mlp_projs = get_output_projections(model, layer_idx)[:1], get_output_projections(model, layer_idx)[1:]
    else:
        attn_projs, mlp_projs = get_input_projections(model, layer_idx)[:3], get_input_projections(model, layer_idx)[3:]
    projections = {"attn": attn_projs, "mlp": mlp_projs, "both": attn_projs + mlp_projs}[targets]

    for proj in projections:
        W = proj.weight.data.float()
        b = (
            proj.bias.data.float() if proj.bias is not None
            else torch.zeros(proj.out_features, device=W.device)
        )

        U_, V_ = U.to(W.device), V.to(W.device)
        if mode == "output":
            # (I + alpha S) W with S = U^T V + V^T U, computed in O(D^2 R)
            W_new = W + alpha * (U_.t() @ (V_ @ W) + V_.t() @ (U_ @ W))
            delta_b = alpha * (U_.t() @ (V_ @ b) + V_.t() @ (U_ @ b))
            if w_p is not None:
                delta_b = delta_b + alpha * w_p.to(W.device)
        else:
            # W (I + alpha S)
            W_new = W + alpha * ((W @ U_.t()) @ V_ + (W @ V_.t()) @ U_)
            delta_b = alpha * (W @ w_p.to(W.device)) if w_p is not None else torch.zeros(proj.out_features, device=W.device)

        proj.weight.data = W_new.to(proj.weight.dtype)
        _add_bias(proj, delta_b)


def absorb_diagonal_into_rmsnorm(
    gamma: torch.Tensor, v: torch.Tensor, std_: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gamma = gamma.detach().float().cpu()
    v = v.detach().float().cpu()
    if std_ is not None:
        v = v / (std_.detach().float().cpu() + 1e-8)
    return gamma * (1.0 + v)


def write_absorbed_rmsnorm(
    model, layer_idx: int, v: torch.Tensor, alpha: float = 1.0,
    std_: Optional[torch.Tensor] = None, norm_name: str = "post_attention_layernorm",
) -> None:
    norm_module = get_norm_module(model, layer_idx, norm_name)
    gamma = norm_module.weight.detach()
    gamma_hat = absorb_diagonal_into_rmsnorm(gamma, alpha * v.to(gamma.dtype), std_=std_)
    with torch.no_grad():
        norm_module.weight.copy_(gamma_hat.to(gamma.device, gamma.dtype))


_BIAS_FLAGS = {
    "attention_bias": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
    "mlp_bias": ("mlp", ("gate_proj", "up_proj", "down_proj")),
}


def _materialize_absorbed_biases(model) -> None:
    """Make biases added by weight absorption survive save_pretrained/from_pretrained.

    Llama-family configs declare projections without bias, so a bias added to one
    projection would be dropped on reload. Give every projection in the group a
    (zero) bias and turn on the matching config flag.
    """
    config = getattr(model.config, "text_config", model.config)
    for flag, (parent_name, proj_names) in _BIAS_FLAGS.items():
        projs = [
            getattr(getattr(layer, parent_name), name)
            for layer in get_layers(model) for name in proj_names
        ]
        if not any(p.bias is not None for p in projs) or getattr(config, flag, False):
            continue
        if not hasattr(config, flag):
            warnings.warn(
                f"{type(config).__name__} has no '{flag}' option; absorbed biases in "
                f"{parent_name} projections will be lost when the checkpoint is reloaded."
            )
            continue
        for p in projs:
            if p.bias is None:
                p.bias = torch.nn.Parameter(
                    torch.zeros(p.out_features, dtype=p.weight.dtype, device=p.weight.device)
                )
        setattr(config, flag, True)


def save_model(model, tokenizer, save_path: str, copy_first: bool = True):
    if copy_first:
        model = copy.deepcopy(model)
    _materialize_absorbed_biases(model)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return model


class TaskVector:

    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        if vector is not None:
            self.vector = vector
            return
        assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
        from transformers import AutoModelForCausalLM

        with torch.no_grad():
            if isinstance(pretrained_checkpoint, str):
                pretrained_checkpoint = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
            if isinstance(finetuned_checkpoint, str):
                finetuned_checkpoint = AutoModelForCausalLM.from_pretrained(finetuned_checkpoint)
            pre_sd = pretrained_checkpoint.state_dict()
            ft_sd = finetuned_checkpoint.state_dict()
            self.vector = {
                k: ft_sd[k] - pre_sd[k]
                for k in pre_sd
                if pre_sd[k].dtype not in (torch.int64, torch.uint8)
            }

    def __add__(self, other: "TaskVector") -> "TaskVector":
        keys = set(self.vector) | set(other.vector)
        return TaskVector(vector={
            k: self.vector.get(k, 0) + other.vector.get(k, 0) for k in keys
        })

    def __neg__(self) -> "TaskVector":
        return TaskVector(vector={k: -v for k, v in self.vector.items()})

    def __mul__(self, scalar: float) -> "TaskVector":
        return TaskVector(vector={k: scalar * v for k, v in self.vector.items()})

    __rmul__ = __mul__

    def apply_to(self, pretrained_checkpoint, scaling_coef: float = 1.0):
        from transformers import AutoModelForCausalLM

        with torch.no_grad():
            if isinstance(pretrained_checkpoint, str):
                model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
            else:
                model = pretrained_checkpoint
            pre_sd = model.state_dict()
            new_sd = {
                k: pre_sd[k] + scaling_coef * self.vector[k] if k in self.vector else pre_sd[k]
                for k in pre_sd
            }
        model.load_state_dict(new_sd, strict=False)
        return model
