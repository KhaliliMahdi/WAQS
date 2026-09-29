"""Weight absorption must reproduce explicit (hook-based) affine steering exactly.

Covers the three absorption points of the paper's Figure 2:
  (a) after a linear projection   -> inject_quadratic_probe(mode="output")
  (b) before a linear projection  -> inject_quadratic_probe(mode="input")
  (c) through the RMSNorm scale   -> write_absorbed_rmsnorm
"""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from quadprobe import inject_quadratic_probe, write_absorbed_rmsnorm
from quadprobe.model_utils import get_input_projections, get_layer, get_output_projections

HIDDEN, RANK, LAYER, ALPHA = 64, 4, 1, 0.05


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    V = 0.1 * torch.randn(RANK, HIDDEN)
    w_p = 0.1 * torch.randn(HIDDEN)
    input_ids = torch.randint(0, 128, (2, 16))
    return model, V, w_p, input_ids


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits


def _affine(x, V, w_p):
    # T(x) = (I + 2 alpha V^T V) x + alpha w_p
    return x + 2 * ALPHA * (x @ V.T) @ V + ALPHA * w_p


def test_absorb_after_projection(setup):
    model, V, w_p, input_ids = setup
    hooked, absorbed = copy.deepcopy(model), copy.deepcopy(model)

    handles = [
        proj.register_forward_hook(lambda m, i, out: _affine(out, V, w_p))
        for proj in get_output_projections(hooked, LAYER)
    ]
    expected = _logits(hooked, input_ids)
    for h in handles:
        h.remove()

    inject_quadratic_probe(absorbed, LAYER, V, w_p=w_p, alpha=ALPHA, mode="output")
    torch.testing.assert_close(_logits(absorbed, input_ids), expected, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(expected, _logits(model, input_ids)), "steering had no effect"


def test_absorb_before_projection(setup):
    model, V, w_p, input_ids = setup
    hooked, absorbed = copy.deepcopy(model), copy.deepcopy(model)

    handles = [
        proj.register_forward_pre_hook(lambda m, args: (_affine(args[0], V, w_p),))
        for proj in get_input_projections(hooked, LAYER)
    ]
    expected = _logits(hooked, input_ids)
    for h in handles:
        h.remove()

    inject_quadratic_probe(absorbed, LAYER, V, w_p=w_p, alpha=ALPHA, mode="input")
    torch.testing.assert_close(_logits(absorbed, input_ids), expected, rtol=1e-4, atol=1e-4)


def test_absorb_through_rmsnorm(setup):
    model, _, _, input_ids = setup
    hooked, absorbed = copy.deepcopy(model), copy.deepcopy(model)
    v = 0.1 * torch.randn(HIDDEN)

    norm = get_layer(hooked, LAYER).post_attention_layernorm
    h = norm.register_forward_hook(lambda m, i, out: out * (1.0 + ALPHA * v))
    expected = _logits(hooked, input_ids)
    h.remove()

    write_absorbed_rmsnorm(absorbed, LAYER, v, alpha=ALPHA)
    torch.testing.assert_close(_logits(absorbed, input_ids), expected, rtol=1e-4, atol=1e-4)


def test_absorbed_checkpoint_roundtrip(setup, tmp_path):
    """An absorbed model saved with save_model reloads as a plain HF checkpoint with identical outputs."""
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models

    from quadprobe import save_model

    model, V, w_p, input_ids = setup
    absorbed = copy.deepcopy(model)
    inject_quadratic_probe(absorbed, LAYER, V, w_p=w_p, alpha=ALPHA, mode="output")
    expected = _logits(absorbed, input_ids)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    save_model(absorbed, tokenizer, str(tmp_path))
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path)).eval()
    torch.testing.assert_close(_logits(reloaded, input_ids), expected, rtol=1e-4, atol=1e-4)


def test_absorbed_probe_matches_autograd_steering(setup):
    """Absorbing a trained (standardized, U != V) probe realizes x + alpha * grad_x f(x) exactly."""
    from quadprobe import QuadraticProbe

    model, _, _, input_ids = setup
    torch.manual_seed(1)
    probe = QuadraticProbe(HIDDEN, rank=RANK)
    with torch.no_grad():
        probe.mean_.copy_(torch.randn(HIDDEN))
        probe.std_.copy_(torch.rand(HIDDEN) + 0.5)

    def steer(out):
        with torch.enable_grad():
            x = out.detach().requires_grad_(True)
            grad = torch.autograd.grad(probe(x).sum(), x)[0]
        return out + ALPHA * grad

    hooked, absorbed = copy.deepcopy(model), copy.deepcopy(model)
    o_proj = get_layer(hooked, LAYER).self_attn.o_proj
    h = o_proj.register_forward_hook(lambda m, i, out: steer(out))
    expected = _logits(hooked, input_ids)
    h.remove()

    U_raw, V_raw, w_raw = probe.raw_space_params()
    inject_quadratic_probe(absorbed, LAYER, V_raw, w_p=w_raw, alpha=ALPHA, U=U_raw, targets="attn")
    torch.testing.assert_close(_logits(absorbed, input_ids), expected, rtol=1e-4, atol=1e-4)
