from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_utils import get_layers, get_norm_module


def _last_token_indices(attn_mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    if padding_side == "left":
        return torch.full((attn_mask.shape[0],), attn_mask.shape[1] - 1, dtype=torch.long)
    return attn_mask.sum(dim=1) - 1


def load_model(model_path: str, device: str = "cuda", dtype=torch.float16) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto" if device == "cuda" else device,
    )
    model.eval()
    return model, tokenizer


class ActivationExtractor:
    def __init__(self, model, tokenizer, layers: List[int], device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device

    @torch.no_grad()
    def extract(
        self,
        sentences: List[str],
        token_strategy: str = "last",
        target_indices: Optional[List[int]] = None,
        batch_size: int = 8,
        max_length: int = 128,
    ) -> Dict[int, np.ndarray]:
        all_acts = {l: [] for l in self.layers}

        for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting activations"):
            batch_sents = sentences[i:i + batch_size]
            batch_tgt = target_indices[i:i + batch_size] if target_indices is not None else None

            enc = self.tokenizer(
                batch_sents, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(self.device)

            outputs = self.model(**enc, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            attn_mask = enc["attention_mask"].cpu()

            for layer_idx in self.layers:
                acts = hidden_states[layer_idx + 1].float().cpu()

                if token_strategy == "last":
                    lengths = _last_token_indices(attn_mask, self.tokenizer.padding_side)
                    vecs = acts[torch.arange(len(batch_sents)), lengths]
                elif token_strategy == "mean":
                    mask = attn_mask.unsqueeze(-1).float()
                    vecs = (acts * mask).sum(1) / mask.sum(1)
                elif token_strategy == "target_idx":
                    assert batch_tgt is not None, "target_indices required for token_strategy='target_idx'"
                    vecs = acts[torch.arange(len(batch_sents)), batch_tgt]
                else:
                    raise ValueError(f"Unknown token_strategy: {token_strategy}")

                all_acts[layer_idx].append(vecs.numpy())

        return {l: np.concatenate(v, axis=0) for l, v in all_acts.items()}


class NormActivationExtractor:

    def __init__(
        self, model, tokenizer, norm_name: str = "post_attention_layernorm",
        layers: Optional[List[int]] = None, device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.norm_name = norm_name
        num_layers = len(get_layers(model))
        self.layers = layers if layers is not None else list(range(num_layers))
        self.device = device

    @torch.no_grad()
    def extract(
        self,
        sentences: List[str],
        token_strategy: str = "last",
        batch_size: int = 8,
        max_length: int = 128,
    ) -> Dict[int, np.ndarray]:
        all_acts = {l: [] for l in self.layers}
        captured: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def _hook(module, inp, out):
                captured[layer_idx] = out.detach()
            return _hook

        handles = [
            get_norm_module(self.model, l, self.norm_name).register_forward_hook(make_hook(l))
            for l in self.layers
        ]

        try:
            for i in tqdm(range(0, len(sentences), batch_size), desc=f"Extracting {self.norm_name}"):
                batch_sents = sentences[i:i + batch_size]
                enc = self.tokenizer(
                    batch_sents, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)
                attn_mask = enc["attention_mask"].cpu()

                captured.clear()
                self.model(**enc)

                for layer_idx in self.layers:
                    acts = captured[layer_idx].float().cpu()

                    if token_strategy == "last":
                        lengths = _last_token_indices(attn_mask, self.tokenizer.padding_side)
                        vecs = acts[torch.arange(len(batch_sents)), lengths]
                    elif token_strategy == "mean":
                        mask = attn_mask.unsqueeze(-1).float()
                        vecs = (acts * mask).sum(1) / mask.sum(1)
                    else:
                        raise ValueError(f"Unknown token_strategy: {token_strategy}")

                    all_acts[layer_idx].append(vecs.numpy())
        finally:
            for h in handles:
                h.remove()

        return {l: np.concatenate(v, axis=0) for l, v in all_acts.items()}


POSITIONS = (
    "residual_pre_attn",
    "input_layernorm_out",
    "attn_output",
    "residual_mid",
    "post_attention_layernorm_out",
    "mlp_output",
    "residual_out",
)


class MultiPointActivationExtractor:

    def __init__(self, model, tokenizer, layers: Optional[List[int]] = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        num_layers = len(get_layers(model))
        self.layers = layers if layers is not None else list(range(num_layers))
        self.device = device

    @torch.no_grad()
    def extract(
        self,
        sentences: List[str],
        token_strategy: str = "last",
        batch_size: int = 8,
        max_length: int = 128,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        all_acts = {pos: {l: [] for l in self.layers} for pos in POSITIONS}
        captured: Dict[str, Dict[int, torch.Tensor]] = {
            pos: {} for pos in POSITIONS if pos not in ("residual_pre_attn", "residual_out")
        }

        def make_fwd_hook(pos, layer_idx):
            def _hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                captured[pos][layer_idx] = t.detach()
            return _hook

        def make_pre_hook(pos, layer_idx):
            def _hook(module, inp):
                captured[pos][layer_idx] = inp[0].detach()
            return _hook

        handles = []
        from .model_utils import get_layer
        for l in self.layers:
            layer = get_layer(self.model, l)
            handles.append(layer.input_layernorm.register_forward_hook(make_fwd_hook("input_layernorm_out", l)))
            handles.append(layer.self_attn.register_forward_hook(make_fwd_hook("attn_output", l)))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(make_pre_hook("residual_mid", l)))
            handles.append(layer.post_attention_layernorm.register_forward_hook(make_fwd_hook("post_attention_layernorm_out", l)))
            handles.append(layer.mlp.register_forward_hook(make_fwd_hook("mlp_output", l)))

        try:
            for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting all positions"):
                batch_sents = sentences[i:i + batch_size]
                enc = self.tokenizer(
                    batch_sents, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)
                attn_mask = enc["attention_mask"].cpu()
                lengths = _last_token_indices(attn_mask, self.tokenizer.padding_side)
                batch_ar = torch.arange(len(batch_sents))

                for pos in captured:
                    captured[pos].clear()

                outputs = self.model(**enc, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                def pool(acts_bsd: torch.Tensor) -> np.ndarray:
                    acts_bsd = acts_bsd.float().cpu()
                    if token_strategy == "last":
                        vecs = acts_bsd[batch_ar, lengths]
                    elif token_strategy == "mean":
                        mask = attn_mask.unsqueeze(-1).float()
                        vecs = (acts_bsd * mask).sum(1) / mask.sum(1)
                    else:
                        raise ValueError(f"Unknown token_strategy: {token_strategy}")
                    return vecs.numpy()

                for layer_idx in self.layers:
                    all_acts["residual_pre_attn"][layer_idx].append(pool(hidden_states[layer_idx]))
                    all_acts["residual_out"][layer_idx].append(pool(hidden_states[layer_idx + 1]))
                    for pos in captured:
                        all_acts[pos][layer_idx].append(pool(captured[pos][layer_idx]))
        finally:
            for h in handles:
                h.remove()

        return {
            pos: {l: np.concatenate(v, axis=0) for l, v in layer_map.items()}
            for pos, layer_map in all_acts.items()
        }


def save_activations(acts: Dict[int, np.ndarray], path: str) -> None:
    np.savez(path, **{f"layer_{k}": v for k, v in acts.items()})


def load_activations(path: str) -> Dict[int, np.ndarray]:
    data = np.load(path)
    return {int(k.replace("layer_", "")): data[k] for k in data.files}
