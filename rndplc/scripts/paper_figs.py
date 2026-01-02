from __future__ import annotations

import argparse
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from retnet_mlx.config import DecayConfig, ModelConfig
from retnet_mlx.decay_policy import FixedDecayPolicy, LearnedStaticDecayPolicy
from retnet_mlx.model import RetNetLM
from retnet_mlx.ops import build_decay_mask_from_gamma
from retnet_mlx.retention import (
    chunkwise_retention,
    parallel_retention_fast_fixed,
    parallel_retention_general,
    recurrent_retention_step,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for figure export. Run: uv add matplotlib"
    ) from exc


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


def get_batch(
    data: np.ndarray, batch_size: int, seq_len: int
) -> tuple[mx.array, mx.array]:
    max_start = len(data) - seq_len - 1
    idx = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data[i : i + seq_len] for i in idx], axis=0)
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in idx], axis=0)
    return mx.array(x), mx.array(y)


def train_with_loss(
    model: RetNetLM,
    data: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    clip_norm: float,
    mode: str,
    chunk_size: int,
) -> list[float]:
    optimizer = optim.AdamW(learning_rate=lr)

    def loss_fn(x: mx.array, y: mx.array) -> mx.array:
        logits = model(x, mode=mode, chunk_size=chunk_size)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        loss = nn.losses.cross_entropy(logits, y)
        loss = mx.mean(loss) + model.decay_reg_loss()
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    for _ in range(steps):
        x, y = get_batch(data, batch_size, seq_len)
        loss, grads = loss_and_grad(x, y)
        grads = optim.clip_grad_norm(grads, clip_norm)[0]
        optimizer.update(model, grads)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return losses


def generate_sample(
    model: RetNetLM,
    *,
    prompt: str,
    stoi: dict[str, int],
    itos: list[str],
    max_new_tokens: int,
    top_k: int,
) -> str:
    prompt_tokens = encode(prompt, stoi)
    cache = None
    for token in prompt_tokens:
        _, cache = model.forward_step(mx.array([token]), cache)

    generated = list(prompt_tokens)
    last_token = prompt_tokens[-1] if prompt_tokens else 0
    for _ in range(max_new_tokens):
        logits, cache = model.forward_step(mx.array([last_token]), cache)
        if top_k <= 0:
            next_token = int(mx.argmax(logits, axis=-1).item())
        else:
            values, indices = mx.topk(logits, k=top_k)
            probs = nn.softmax(values, axis=-1)
            choice = mx.random.categorical(probs)
            next_token = int(indices[0, choice.item()].item())
        generated.append(next_token)
        last_token = next_token

    return decode(generated, itos)


def bench_parallel(model: RetNetLM, tokens: mx.array, steps: int) -> float:
    for _ in range(2):
        out = model(tokens, mode="parallel")
        mx.eval(out)
    start = time.perf_counter()
    for _ in range(steps):
        out = model(tokens, mode="parallel")
        mx.eval(out)
    elapsed = time.perf_counter() - start
    return steps * tokens.size / elapsed


def bench_chunkwise(
    model: RetNetLM, tokens: mx.array, steps: int, chunk_size: int
) -> float:
    for _ in range(2):
        out = model(tokens, mode="chunkwise", chunk_size=chunk_size)
        mx.eval(out)
    start = time.perf_counter()
    for _ in range(steps):
        out = model(tokens, mode="chunkwise", chunk_size=chunk_size)
        mx.eval(out)
    elapsed = time.perf_counter() - start
    return steps * tokens.size / elapsed


def bench_recurrent(model: RetNetLM, tokens: mx.array, steps: int) -> float:
    for _ in range(2):
        cache = None
        for idx in range(tokens.shape[1]):
            logits, cache = model.forward_step(tokens[:, idx], cache)
            mx.eval(logits)
    start = time.perf_counter()
    total_tokens = steps * tokens.size
    for _ in range(steps):
        cache = None
        for idx in range(tokens.shape[1]):
            logits, cache = model.forward_step(tokens[:, idx], cache)
            mx.eval(logits)
    elapsed = time.perf_counter() - start
    return total_tokens / elapsed


def _rollout_recurrent(q, k, v, alpha):
    b, h, t, d = q.shape
    dv = v.shape[-1]
    past = mx.zeros((b, h, d, dv), dtype=mx.float32)
    outs = []
    for idx in range(t):
        alpha_t = alpha if alpha.ndim <= 1 else alpha[:, :, idx]
        out_t, past = recurrent_retention_step(
            q[:, :, idx], k[:, :, idx], v[:, :, idx], past, alpha_t
        )
        outs.append(out_t)
    return mx.stack(outs, axis=2)


def _rollout_chunkwise(q, k, v, alpha, chunk_size):
    b, h, t, _ = q.shape
    past = None
    outs = []
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        q_chunk = q[:, :, start:end]
        k_chunk = k[:, :, start:end]
        v_chunk = v[:, :, start:end]
        if alpha.ndim <= 1:
            alpha_chunk = mx.broadcast_to(alpha[None, :, None], (b, h, end - start))
        else:
            alpha_chunk = alpha[:, :, start:end]
        out_chunk, past = chunkwise_retention(
            q_chunk, k_chunk, v_chunk, past, alpha_chunk
        )
        outs.append(out_chunk)
    return mx.concatenate(outs, axis=2)


def compute_equivalence_metrics() -> dict[str, dict[str, float]]:
    mx.random.seed(0)
    b, h, t, dqk, dv = 2, 3, 16, 8, 16
    q = mx.random.normal((b, h, t, dqk))
    k = mx.random.normal((b, h, t, dqk))
    v = mx.random.normal((b, h, t, dv))

    metrics: dict[str, dict[str, float]] = {}

    for name, policy in (
        ("fixed", FixedDecayPolicy(DecayConfig(policy="fixed"), h, dqk)),
        (
            "learned_static",
            LearnedStaticDecayPolicy(DecayConfig(policy="learned_static"), h, dqk),
        ),
    ):
        alpha = policy.alpha(mx.zeros((b, t, dqk)), "parallel", layer_idx=0)
        decay_mask = build_decay_mask_from_gamma(alpha, t)
        out_parallel = parallel_retention_fast_fixed(q, k, v, decay_mask)
        out_recurrent = _rollout_recurrent(q, k, v, alpha)
        rec_err = mx.max(mx.abs(out_parallel - out_recurrent)).item()

        chunk_errs = []
        for chunk_size in (4, 8):
            out_chunk = _rollout_chunkwise(q, k, v, alpha, chunk_size)
            chunk_errs.append(mx.max(mx.abs(out_parallel - out_chunk)).item())

        metrics[name] = {
            "recurrent_max_abs": float(rec_err),
            "chunkwise_max_abs": float(max(chunk_errs)),
        }

    alpha = mx.random.uniform(low=0.05, high=0.95, shape=(b, h, t))
    out_parallel = parallel_retention_general(q, k, v, alpha)
    out_recurrent = _rollout_recurrent(q, k, v, alpha)
    metrics["general"] = {
        "recurrent_max_abs": float(mx.max(mx.abs(out_parallel - out_recurrent)).item())
    }

    return metrics


def plot_loss(losses: list[float], out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#1f77b4")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_throughput(throughput: dict[str, float], out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(throughput.keys())
    values = [throughput[k] for k in labels]
    ax.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_equivalence(
    metrics: dict[str, dict[str, float]], out_path: pathlib.Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    labels = ["fixed", "learned_static", "general"]
    rec_vals = [metrics[name]["recurrent_max_abs"] for name in labels]
    axes[0].bar(labels, rec_vals, color="#1f77b4")
    axes[0].set_title("Recurrent vs Parallel")
    axes[0].set_ylabel("Max abs error")

    labels_chunk = ["fixed", "learned_static"]
    chunk_vals = [metrics[name]["chunkwise_max_abs"] for name in labels_chunk]
    axes[1].bar(labels_chunk, chunk_vals, color="#2ca02c")
    axes[1].set_title("Chunkwise vs Parallel")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gammas(out_path: pathlib.Path, n_heads: int, d_model: int) -> None:
    fixed = FixedDecayPolicy(DecayConfig(policy="fixed"), n_heads, d_model)
    learned = LearnedStaticDecayPolicy(
        DecayConfig(policy="learned_static", monotonic_heads=True), n_heads, d_model
    )
    gamma_fixed = np.asarray(fixed.gamma())
    gamma_learned = np.asarray(learned.gamma())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gamma_fixed, marker="o", label="fixed")
    ax.plot(gamma_learned, marker="o", label="learned_static")
    ax.set_xlabel("Head")
    ax.set_ylabel("Gamma")
    ax.set_title("Per-head Decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample(sample: str, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.01, 0.99, sample, ha="left", va="top", family="monospace", wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/tiny.txt")
    parser.add_argument("--out_dir", type=str, default="artifacts/paper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--bench_steps", type=int, default=10)
    parser.add_argument("--bench_batch", type=int, default=1)
    parser.add_argument("--bench_seq_len", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_qk", type=int, default=32)
    parser.add_argument("--d_v", type=int, default=64)
    parser.add_argument("--ffn_mult", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    data_path = pathlib.Path(args.data)
    text = load_text(data_path)
    stoi, itos = build_vocab(text)
    data = np.array(encode(text, stoi), dtype=np.int32)

    decay = DecayConfig(policy="learned_static", monotonic_heads=True)
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

    losses = train_with_loss(
        model,
        data,
        steps=args.train_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        clip_norm=args.clip_norm,
        mode="parallel",
        chunk_size=args.chunk_size,
    )

    sample = generate_sample(
        model,
        prompt=args.prompt,
        stoi=stoi,
        itos=itos,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
    )

    tokens = mx.random.randint(
        0, len(itos), shape=(args.bench_batch, args.bench_seq_len)
    )
    throughput = {
        "parallel": bench_parallel(model, tokens, args.bench_steps),
        "chunkwise": bench_chunkwise(model, tokens, args.bench_steps, args.chunk_size),
        "recurrent": bench_recurrent(model, tokens, args.bench_steps),
    }

    equivalence = compute_equivalence_metrics()

    out_dir = pathlib.Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    plot_loss(losses, fig_dir / "loss_curve.png")
    plot_throughput(throughput, fig_dir / "throughput.png")
    plot_equivalence(equivalence, fig_dir / "equivalence.png")
    plot_gammas(fig_dir / "gamma_curves.png", args.n_heads, args.d_model)
    plot_sample(sample, fig_dir / "sample_generation.png")

    with (data_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "losses": losses,
                "throughput": throughput,
                "equivalence": equivalence,
                "sample": sample,
                "config": {
                    "vocab_size": len(itos),
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "d_qk": args.d_qk,
                    "d_v": args.d_v,
                    "ffn_mult": args.ffn_mult,
                    "train_steps": args.train_steps,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                },
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
