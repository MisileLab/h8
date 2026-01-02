from __future__ import annotations

import argparse
import pathlib

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import trange

from retnet_mlx.config import DecayConfig, ModelConfig
from retnet_mlx.model import RetNetLM


def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = [ch for ch in chars]
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi[ch] for ch in text], dtype=np.int32)


def get_batch(
    data: np.ndarray, batch_size: int, seq_len: int
) -> tuple[mx.array, mx.array]:
    max_start = len(data) - seq_len - 1
    idx = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data[i : i + seq_len] for i in idx], axis=0)
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in idx], axis=0)
    return mx.array(x), mx.array(y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/tiny.txt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--mode", choices=["parallel", "chunkwise"], default="parallel")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--decay_policy",
        choices=["fixed", "learned_static", "conditional"],
        default="learned_static",
    )
    parser.add_argument("--gamma_min", type=float, default=0.1)
    parser.add_argument("--gamma_max", type=float, default=0.999)
    parser.add_argument("--decay_reg_strength", type=float, default=0.0)
    parser.add_argument("--monotonic_heads", action="store_true", default=True)
    parser.add_argument(
        "--no_monotonic_heads", action="store_false", dest="monotonic_heads"
    )
    parser.add_argument("--h_min", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_qk", type=int, default=32)
    parser.add_argument("--d_v", type=int, default=64)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="checkpoints/char_lm.npz")
    args = parser.parse_args()

    np.random.seed(0)
    mx.random.seed(0)

    data_path = pathlib.Path(args.data)
    text = load_text(data_path)
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    decay = DecayConfig(
        policy=args.decay_policy,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        monotonic_heads=args.monotonic_heads,
        h_min=args.h_min,
        decay_reg_strength=args.decay_reg_strength,
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
    optimizer = optim.AdamW(learning_rate=args.lr)

    def loss_fn(x: mx.array, y: mx.array) -> mx.array:
        logits = model(x, mode=args.mode, chunk_size=args.chunk_size)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        loss = nn.losses.cross_entropy(logits, y)
        loss = mx.mean(loss) + model.decay_reg_loss()
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in trange(args.steps):
        x, y = get_batch(data, args.batch_size, args.seq_len)
        loss, grads = loss_and_grad(x, y)
        clipped = optim.clip_grad_norm(grads, args.clip_norm)
        grads = clipped[0] if isinstance(clipped, tuple) else clipped
        optimizer.update(model, grads)
        mx.eval(loss)
        if (step + 1) % 50 == 0:
            print(f"step {step + 1} loss {loss.item():.4f}")

    save_path = pathlib.Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(save_path))


if __name__ == "__main__":
    main()
