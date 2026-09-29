# WAQS: Weight Absorption for Quadratic Probing and Affine Steering

**Ding Zhu, Mahdi Khalili**

*Under review* · [Project page](https://khalilimahdi.github.io/publication/waqs)

WAQS gives LLM activation steering that adapts to each input and adds no inference cost. A concept (e.g. harmful vs. harmless) is modeled with a quadratic probe f(x) = xᵀW_p x + w_pᵀx + b. Its gradient step, T(x) = x + α∇f(x) = (I + 2αW_p)x + αw_p, is affine in the activation, so it can be folded exactly into the adjacent linear layers with an offline weight update. The result is a standard checkpoint: no hooks, no custom inference code.

![Weight absorption](https://khalilimahdi.github.io/images/publications/waqs-absorption.png)

## Installation

```bash
git clone https://github.com/KhaliliMahdi/WAQS.git
cd WAQS
pip install torch  # choose the build for your CUDA version from https://pytorch.org
pip install -r requirements.txt
```

`quadprobe` supports Llama-family architectures (Llama 2/3, Mistral, Qwen2, SmolLM, and Gemma 3 through its language model).

## Quick start

`examples/refusal_steering.py` runs the full pipeline on a small model in a few minutes on CPU:

```bash
PYTHONPATH=. python examples/refusal_steering.py --model HuggingFaceTB/SmolLM2-135M-Instruct --layer 15
```

It:

1. Extracts last-token attention-output activations for harmful ([AdvBench](https://github.com/llm-attacks/llm-attacks)) and harmless ([Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)) prompts.
2. Fits a linear probe and rank-4 quadratic probes, using shrinkage GDA and logistic regression.
3. Absorbs the quadratic steering map into the layer's `o_proj`.
4. Saves the model, reloads it with plain `transformers`, and reports how far the probe score of held-out prompts moved.

Example output:

```
linear probe                     test acc 0.990  AUROC 1.000
GDA quadratic probe (rank 4)     test acc 0.505  AUROC 0.553
LR quadratic probe (rank 4)      test acc 0.971  AUROC 0.999

Absorbing QuadraticProbe into layer 15 o_proj with alpha=3.95
mean probe logit on 57 held-out harmless prompts: -8.57 (base) -> -0.42 (absorbed checkpoint)
```

Useful options: `--model`, `--layer`, `--rank`, `--strength` (steering size as a fraction of the mean activation norm; negative values suppress the concept), `--n` (prompts per class), `--output-dir`.

## Using the library

```python
from quadprobe import (MultiPointActivationExtractor, QuadraticProbeTrainer,
                       inject_quadratic_probe, load_model, save_model)

model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf")
layer = 14

# 1. Activations at the attention output of one layer, for positive (y=1) and negative (y=0) prompts
acts = MultiPointActivationExtractor(model, tokenizer, layers=[layer]).extract(prompts)
X = acts["attn_output"][layer]

# 2. Low-rank quadratic probe
trainer = QuadraticProbeTrainer(X.shape[1], rank=4, device="cuda")
trainer.fit(X_train, y_train, X_val, y_val)
probe = trainer.probe.cpu()

# 3. Absorb T(x) = x + alpha * grad f(x) into o_proj (raw_space_params undoes the probe's standardization)
U, V, w = probe.raw_space_params()
inject_quadratic_probe(model, layer, V, w_p=w, alpha=1.0, U=U, targets="attn")

# 4. A standard checkpoint that loads with AutoModelForCausalLM.from_pretrained
save_model(model, tokenizer, "llama2-7b-waqs")
```

Absorption points (Figure 2 of the paper):

| Where | Call |
|---|---|
| (a) after a linear projection (`o_proj`, `down_proj`) | `inject_quadratic_probe(..., mode="output")` |
| (b) before a linear projection (`q/k/v_proj`, `gate/up_proj`) | `inject_quadratic_probe(..., mode="input")` |
| (c) through the RMSNorm scale (diagonal probe) | `write_absorbed_rmsnorm(...)` |

`targets="attn" | "mlp" | "both"` selects the attention path, the MLP path, or both.

The package also includes the baselines used in the paper: `LinearProbe`, difference-in-means (`compute_mean_diff`), `SteeringHook` (constant vectors), `AngularSteeringHook`, `SphericalSteeringHook`, and `DynamicQuadraticSteeringHook` (hook-based, non-absorbed quadratic steering).

## Tests

```bash
PYTHONPATH=. pytest tests
```

The tests use a small random Llama model and check that:
* each absorption point matches the explicit hook-based steering;
* absorbing a trained probe reproduces x + α∇f(x), with the gradient computed by autograd through the probe;
* the absorbed model reloads from disk with identical outputs.

## Citation

```bibtex
@misc{zhu2026waqs,
  title  = {{WAQS}: Weight Absorption for Quadratic Probing and Affine Steering},
  author = {Zhu, Ding and Khalili, Mohammad Mahdi},
  year   = {2026},
  note   = {Under review}
}
```
