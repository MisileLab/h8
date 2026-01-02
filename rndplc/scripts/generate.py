from __future__ import annotations

import argparse
import pathlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from retnet_mlx.config import DecayConfig, ModelConfig
from retnet_mlx.model import RetNetLM


def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = [ch for ch in chars]
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(tokens: list[int], itos: list[str]) -> str:
    return "".join(itos[t] for t in tokens)


def sample_logits(logits: mx.array, top_k: int) -> int:
    if top_k <= 0:
        return int(mx.argmax(logits, axis=-1).item())

    values, indices = mx.topk(logits, k=top_k)
    probs = nn.softmax(values, axis=-1)
    choice = mx.random.categorical(probs)
    return int(indices[0, choice.item()].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/tiny.txt")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/char_lm.npz")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_qk", type=int, default=32)
    parser.add_argument("--d_v", type=int, default=64)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument(
        "--decay_policy",
        choices=["fixed", "learned_static", "conditional"],
        default="learned_static",
    )
    parser.add_argument("--gamma_min", type=float, default=0.1)
    parser.add_argument("--gamma_max", type=float, default=0.999)
    parser.add_argument("--monotonic_heads", action="store_true", default=True)
    parser.add_argument(
        "--no_monotonic_heads", action="store_false", dest="monotonic_heads"
    )
    parser.add_argument("--h_min", type=float, default=1.0)
    args = parser.parse_args()

    np.random.seed(0)
    mx.random.seed(0)

    data_path = pathlib.Path(args.data)
    text = load_text(data_path)
    stoi, itos = build_vocab(text)

    decay = DecayConfig(
        policy=args.decay_policy,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        monotonic_heads=args.monotonic_heads,
        h_min=args.h_min,
    )
    config = ModelConfig(
        vocab_size=len(itos),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_qk=args.d_qk,
        d_v=args.d_v,
        ffn_mult=args.ffn_mult,
        decay=decay,
    )
    model = RetNetLM(config)

    ckpt = pathlib.Path(args.checkpoint)
    if ckpt.exists():
        model.load_weights(str(ckpt))

    prompt_tokens = encode(args.prompt, stoi)
    cache = None
    for token in prompt_tokens:
        _, cache = model.forward_step(mx.array([token]), cache)

    generated = list(prompt_tokens)
    last_token = prompt_tokens[-1] if prompt_tokens else 0

    for _ in range(args.max_new_tokens):
        logits, cache = model.forward_step(mx.array([last_token]), cache)
        next_token = sample_logits(logits, args.top_k)
        generated.append(next_token)
        last_token = next_token

    print(decode(generated, itos))


if __name__ == "__main__":
    main()
