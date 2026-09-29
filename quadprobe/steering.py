from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from .model_utils import get_layer, get_norm_module


def compute_mean_diff(pos_acts: np.ndarray, neg_acts: np.ndarray, normalize: bool = True) -> np.ndarray:
    vec = pos_acts.mean(0) - neg_acts.mean(0)
    if normalize:
        vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def compute_pca_direction(pos_acts: np.ndarray, neg_acts: np.ndarray, normalize: bool = True) -> np.ndarray:
    diffs = pos_acts - neg_acts[:len(pos_acts)]
    pca = PCA(n_components=1)
    pca.fit(diffs)
    vec = pca.components_[0]
    if normalize:
        vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def get_linear_probe_direction(probe) -> np.ndarray:
    vec = probe.get_weight_vector().detach().cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-8)


def get_quadratic_probe_direction(
    probe, ref_h_norm: torch.Tensor, absorb_alpha: float = 1.0,
) -> np.ndarray:
    with torch.no_grad():
        if ref_h_norm.dim() == 1:
            s = probe.compute_steering_vector(ref_h_norm, alpha=absorb_alpha)
        else:
            s = torch.stack([
                probe.compute_steering_vector(h, alpha=absorb_alpha) for h in ref_h_norm
            ]).mean(0)
    vec = s.detach().cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-8)


def get_quadratic_probe_linear_term(probe) -> np.ndarray:
    vec = probe.w.detach().cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-8)


class SteeringHook:

    def __init__(
        self, model, layer_idx: int, vector: np.ndarray, alpha: float = 1.0,
        token_positions: Union[str, List[int]] = "all", dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.token_positions = token_positions
        self.vec = torch.tensor(vector, dtype=dtype)
        self._handle = None

    def _hook_fn(self, module, inp, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        vec = self.vec.to(hidden.device, hidden.dtype)

        if self.token_positions == "all":
            hidden = hidden + self.alpha * vec
        elif self.token_positions == "last":
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + self.alpha * vec
        elif isinstance(self.token_positions, list):
            hidden = hidden.clone()
            for pos in self.token_positions:
                hidden[:, pos, :] = hidden[:, pos, :] + self.alpha * vec
        else:
            raise ValueError(f"Unknown token_positions: {self.token_positions}")

        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    def __enter__(self) -> "SteeringHook":
        layer = get_layer(self.model, self.layer_idx)
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class DynamicQuadraticSteeringHook:

    def __init__(
        self, model, layer_idx: int, probe, alpha: float = 1.0,
        token_positions: Union[str, List[int]] = "all", dtype: torch.dtype = torch.float16,
        normalize: bool = True,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.probe = probe
        self.alpha = alpha
        self.token_positions = token_positions
        self.dtype = dtype
        self.normalize = normalize
        self._handle = None

    def _hook_fn(self, module, inp, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        device = hidden.device

        mean_ = self.probe.mean_.to(device=device, dtype=torch.float32)
        std_ = self.probe.std_.to(device=device, dtype=torch.float32)
        w = self.probe.w.to(device=device, dtype=torch.float32)
        U = self.probe.U.to(device=device, dtype=torch.float32)
        V = self.probe.V.to(device=device, dtype=torch.float32)

        target = hidden[:, -1:, :] if self.token_positions == "last" else hidden

        h_norm = (target.float() - mean_) / (std_ + 1e-8)
        Uh = h_norm @ U.T
        Vh = h_norm @ V.T
        quad_grad = Uh @ U + Vh @ V
        grad_norm = w + quad_grad

        delta_raw = std_ * grad_norm
        if self.normalize:
            delta_raw = delta_raw / (delta_raw.norm(dim=-1, keepdim=True) + 1e-8)
        delta_raw = (self.alpha * delta_raw).to(self.dtype)

        if self.token_positions == "last":
            hidden = hidden.clone()
            hidden[:, -1:, :] = hidden[:, -1:, :] + delta_raw
        else:
            hidden = hidden + delta_raw

        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    def __enter__(self) -> "DynamicQuadraticSteeringHook":
        layer = get_layer(self.model, self.layer_idx)
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def generate_steered(
    model, tokenizer, prompt: str, layer_idx: int, vector: np.ndarray, alpha: float,
    max_new_tokens: int = 128, device: str = "cuda",
) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with SteeringHook(model, layer_idx, vector, alpha):
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _apply_spherical(h: torch.Tensor, r_unit: torch.Tensor, gamma_t: float, sin_gamma: float) -> torch.Tensor:
    r_ = r_unit.to(h.device)
    norm_h = h.norm(dim=-1, keepdim=True)
    h_unit = h / norm_h.clamp(min=1e-8)
    c = (h_unit * r_).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(c)
    sin_theta = torch.sin(theta).clamp(min=1e-8)
    t = gamma_t
    w_h = torch.sin((1 - t) * theta) / sin_theta
    w_r = torch.sin(t * theta) / sin_theta
    h_new_unit = w_h * h_unit + w_r * r_
    return norm_h * h_new_unit


class SphericalSteeringHook:

    def __init__(self, model, layer_idx: int, r: torch.Tensor, gamma: float = 0.5):
        self.model = model
        self.layer_idx = layer_idx
        self.r_unit = r.float() / r.float().norm()
        self.gamma_t = float(gamma)
        self.sin_gamma = (1.0 - self.gamma_t ** 2) ** 0.5
        self._handles: List = []

    def _pre_hook(self, module, args):
        h = args[0].float()
        h_new = _apply_spherical(h, self.r_unit, self.gamma_t, self.sin_gamma)
        return (h_new.to(args[0].dtype),) + args[1:]

    def _post_hook(self, module, args, output):
        h_new = _apply_spherical(output.float(), self.r_unit, self.gamma_t, self.sin_gamma)
        return h_new.to(output.dtype)

    def _layer_post_hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        h_new = _apply_spherical(hidden.float(), self.r_unit, self.gamma_t, self.sin_gamma).to(hidden.dtype)
        return (h_new,) + tuple(output[1:]) if is_tuple else h_new

    def __enter__(self) -> "SphericalSteeringHook":
        layer = get_layer(self.model, self.layer_idx)
        o_proj_in_features = layer.self_attn.o_proj.weight.shape[1]
        if o_proj_in_features == self.r_unit.shape[0]:
            self._handles = [
                layer.self_attn.o_proj.register_forward_pre_hook(self._pre_hook),
                layer.mlp.down_proj.register_forward_hook(self._post_hook),
            ]
        else:
            self._handles = [layer.register_forward_hook(self._layer_post_hook)]
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles = []


def _apply_angular(h: torch.Tensor, r_unit: torch.Tensor, b2: torch.Tensor, target_dir: torch.Tensor) -> torch.Tensor:
    dev = h.device
    r_, b2_, td_ = r_unit.to(dev), b2.to(dev), target_dir.to(dev)
    c1 = (h * r_).sum(dim=-1, keepdim=True)
    c2 = (h * b2_).sum(dim=-1, keepdim=True)
    proj = c1 * r_ + c2 * b2_
    return h - proj + proj.norm(dim=-1, keepdim=True) * td_


class AngularSteeringHook:

    def __init__(self, model, layer_idx: int, r: torch.Tensor, theta: float = 0.0,
                 b2: Optional[torch.Tensor] = None, seed: int = 42):
        self.model = model
        self.layer_idx = layer_idx
        self.r_unit = r.float() / r.float().norm()

        if b2 is None:
            g = torch.Generator().manual_seed(seed)
            rand_vec = torch.randn(r.shape, generator=g)
            rand_vec = rand_vec - (rand_vec * self.r_unit).sum() * self.r_unit
            self.b2 = rand_vec / rand_vec.norm()
        else:
            b2 = b2.float()
            b2 = b2 - (b2 * self.r_unit).sum() * self.r_unit
            self.b2 = b2 / b2.norm()

        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        self.target_dir = cos_t * self.r_unit + sin_t * self.b2
        self._handles: List = []

    def _pre_hook(self, module, args):
        h = args[0].float()
        h_new = _apply_angular(h, self.r_unit, self.b2, self.target_dir)
        return (h_new.to(args[0].dtype),) + args[1:]

    def _post_hook(self, module, args, output):
        h_new = _apply_angular(output.float(), self.r_unit, self.b2, self.target_dir)
        return h_new.to(output.dtype)

    def _layer_post_hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        h_new = _apply_angular(hidden.float(), self.r_unit, self.b2, self.target_dir).to(hidden.dtype)
        return (h_new,) + tuple(output[1:]) if is_tuple else h_new

    def __enter__(self) -> "AngularSteeringHook":
        layer = get_layer(self.model, self.layer_idx)
        o_proj_in_features = layer.self_attn.o_proj.weight.shape[1]
        if o_proj_in_features == self.r_unit.shape[0]:
            self._handles = [
                layer.self_attn.o_proj.register_forward_pre_hook(self._pre_hook),
                layer.mlp.down_proj.register_forward_hook(self._post_hook),
            ]
        else:
            self._handles = [layer.register_forward_hook(self._layer_post_hook)]
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles = []


class AbsorbedRMSNormSteering:

    def __init__(self, model, layer_idx: int, v: torch.Tensor, alpha: float = 1.0,
                 std_: Optional[torch.Tensor] = None, norm_name: str = "post_attention_layernorm"):
        self.model = model
        self.layer_idx = layer_idx
        self.norm_name = norm_name
        self.alpha = alpha
        self.v = v.detach().float().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        self.std_ = None
        if std_ is not None:
            self.std_ = std_.detach().float().cpu() if isinstance(std_, torch.Tensor) else torch.tensor(std_, dtype=torch.float32)
        self._orig_weight = None
        self._norm_module = None

    def __enter__(self) -> "AbsorbedRMSNormSteering":
        from .writeback import absorb_diagonal_into_rmsnorm

        self._norm_module = get_norm_module(self.model, self.layer_idx, self.norm_name)
        self._orig_weight = self._norm_module.weight.detach().clone()

        gamma = self._orig_weight
        v_scaled = self.alpha * self.v.to(gamma.dtype)
        gamma_hat = absorb_diagonal_into_rmsnorm(gamma, v_scaled, std_=self.std_)
        with torch.no_grad():
            self._norm_module.weight.copy_(gamma_hat.to(gamma.device, gamma.dtype))
        return self

    def __exit__(self, *args):
        if self._norm_module is not None and self._orig_weight is not None:
            with torch.no_grad():
                self._norm_module.weight.copy_(self._orig_weight)
        self._norm_module = None
        self._orig_weight = None
