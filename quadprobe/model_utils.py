from __future__ import annotations

import torch.nn as nn


class UnsupportedArchitectureError(RuntimeError):
    pass


def get_layers(model) -> nn.ModuleList:
    try:
        return model.model.layers
    except AttributeError:
        pass
    try:
        return model.model.language_model.layers
    except AttributeError as e:
        raise UnsupportedArchitectureError(
            f"quadprobe only supports Llama-family architectures "
            f"(model.model.layers[...] or model.model.language_model.layers[...] "
            f"for multimodal wrappers); got {type(model).__name__}."
        ) from e


def get_layer(model, layer_idx: int) -> nn.Module:
    return get_layers(model)[layer_idx]


def get_num_layers(model) -> int:
    return len(get_layers(model))


def get_output_projections(model, layer_idx: int) -> list[nn.Linear]:
    layer = get_layer(model, layer_idx)
    return [layer.self_attn.o_proj, layer.mlp.down_proj]


def get_input_projections(model, layer_idx: int) -> list[nn.Linear]:
    layer = get_layer(model, layer_idx)
    attn, mlp = layer.self_attn, layer.mlp
    return [attn.q_proj, attn.k_proj, attn.v_proj, mlp.gate_proj, mlp.up_proj]


def get_norm_module(model, layer_idx: int, norm_name: str = "post_attention_layernorm") -> nn.Module:
    layer = get_layer(model, layer_idx)
    if not hasattr(layer, norm_name):
        raise UnsupportedArchitectureError(
            f"Layer {layer_idx} ({type(layer).__name__}) has no '{norm_name}' module."
        )
    return getattr(layer, norm_name)


def get_hidden_size(model) -> int:
    text_config = getattr(model.config, "text_config", model.config)
    return text_config.hidden_size
