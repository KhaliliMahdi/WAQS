"""End-to-end WAQS example: quadratic probe for refusal, absorbed into the model weights.

1. Extract last-token attention-output activations of one layer for harmful (AdvBench)
   and harmless (Alpaca) prompts.
2. Fit a linear probe and two low-rank quadratic probes (shrinkage GDA and logistic
   regression) and compare them on held-out prompts.
3. Absorb the quadratic steering map T(x) = x + alpha * grad f(x) into that layer's o_proj.
4. Save the absorbed model as a standard Hugging Face checkpoint, reload it with plain
   transformers, and check that the probe score of held-out prompts moves as intended.

Example:
    python examples/refusal_steering.py --model HuggingFaceTB/SmolLM2-135M-Instruct --layer 15
"""
import argparse
import csv
import io
import urllib.request

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM

from quadprobe import (
    GDAQuadraticProbe,
    LinearProbe,
    MultiPointActivationExtractor,
    QuadraticProbeTrainer,
    inject_quadratic_probe,
    load_model,
    save_model,
)

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def load_prompts(n: int, seed: int):
    with urllib.request.urlopen(ADVBENCH_URL) as f:
        harmful = [row["goal"] for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))]
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"] for r in alpaca if not r["input"].strip()]
    rng = np.random.default_rng(seed)
    harmful = [harmful[i] for i in rng.permutation(len(harmful))[:n]]
    harmless = [harmless[i] for i in rng.permutation(len(harmless))[:n]]
    return harmful, harmless


def chat(tokenizer, prompts):
    if tokenizer.chat_template is None:
        return prompts
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=48):
    enc = tokenizer(chat(tokenizer, [prompt])[0], return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--n", type=int, default=256, help="prompts per class")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--shrinkage", type=float, default=0.5, help="GDA shrinkage toward a scaled identity")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="steering size as a fraction of the mean activation norm (sets alpha)")
    parser.add_argument("--output-dir", default="outputs/absorbed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model, tokenizer = load_model(args.model, device=args.device, dtype=dtype)
    tokenizer.padding_side = "left"

    harmful, harmless = load_prompts(args.n, args.seed)

    def attn_acts(m, prompts):
        extractor = MultiPointActivationExtractor(m, tokenizer, layers=[args.layer], device=args.device)
        return extractor.extract(chat(tokenizer, prompts))["attn_output"][args.layer]

    X_pos, X_neg = attn_acts(model, harmful), attn_acts(model, harmless)
    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(int)

    idx = np.random.default_rng(args.seed).permutation(len(X))
    n_train = int(0.8 * len(X))
    tr, te = idx[:n_train], idx[n_train:]

    linear = LinearProbe().fit(X[tr], y[tr])
    gda = GDAQuadraticProbe.fit_closed_form(
        X[tr], y[tr], rank=args.rank, shrinkage=args.shrinkage, shrink_target="identity")
    trainer = QuadraticProbeTrainer(X.shape[1], rank=args.rank, device=args.device)
    trainer.fit(X[tr], y[tr], X[te], y[te], verbose=False)
    lr_qp = trainer.probe.cpu().eval()
    trainer.device = "cpu"

    scores = {}
    for name, predict in [
        ("linear probe", linear.predict_proba),
        (f"GDA quadratic probe (rank {args.rank})", gda.predict_proba),
        (f"LR quadratic probe (rank {args.rank})", trainer.predict_proba),
    ]:
        proba = predict(X[te])
        acc = ((proba >= 0.5).astype(int) == y[te]).mean()
        scores[name] = roc_auc_score(y[te], proba)
        print(f"{name:32s} test acc {acc:.3f}  AUROC {scores[name]:.3f}")

    probe = lr_qp if scores[f"LR quadratic probe (rank {args.rank})"] >= scores[f"GDA quadratic probe (rank {args.rank})"] else gda
    U, V, w = probe.raw_space_params()

    # Choose alpha so the average steering step is `strength` times the average activation norm.
    Xt = torch.tensor(X[tr], dtype=torch.float32)
    grad = w + Xt @ (U.T @ V + V.T @ U)
    alpha = args.strength * Xt.norm(dim=1).mean().item() / grad.norm(dim=1).mean().item()
    print(f"\nAbsorbing {type(probe).__name__} into layer {args.layer} o_proj with alpha={alpha:.4g}")

    heldout = [harmless[i - len(X_pos)] for i in te if i >= len(X_pos)]
    base_logit = probe(torch.tensor(attn_acts(model, heldout))).mean().item()
    prompt = heldout[0]
    base_reply = generate(model, tokenizer, prompt, args.device)

    inject_quadratic_probe(model, args.layer, V, w_p=w, alpha=alpha, U=U, targets="attn")
    save_model(model, tokenizer, args.output_dir, copy_first=False)

    # The saved checkpoint loads with plain transformers: no hooks or custom code.
    reloaded = AutoModelForCausalLM.from_pretrained(args.output_dir, torch_dtype=dtype).to(args.device).eval()
    steered_logit = probe(torch.tensor(attn_acts(reloaded, heldout))).mean().item()
    print(f"mean probe logit on {len(heldout)} held-out harmless prompts: "
          f"{base_logit:+.2f} (base) -> {steered_logit:+.2f} (absorbed checkpoint)")

    print(f"\nPrompt: {prompt}\nBase model:     {base_reply}")
    print(f"Absorbed model: {generate(reloaded, tokenizer, prompt, args.device)}")
    print(f"\nSaved absorbed checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
