from __future__ import annotations

import argparse
import time

import mlx.core as mx

from retnet_mlx.config import DecayConfig, ModelConfig
from retnet_mlx.model import RetNetLM


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_qk", type=int, default=32)
    parser.add_argument("--d_v", type=int, default=64)
    parser.add_argument("--ffn_mult", type=int, default=4)
    args = parser.parse_args()

    mx.random.seed(0)
    decay = DecayConfig(policy="learned_static")
    config = ModelConfig(
        vocab_size=args.vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_qk=args.d_qk,
        d_v=args.d_v,
        ffn_mult=args.ffn_mult,
        decay=decay,
    )
    model = RetNetLM(config)
    tokens = mx.random.randint(0, args.vocab, shape=(args.batch, args.seq_len))

    tps_parallel = bench_parallel(model, tokens, args.steps)
    tps_chunk = bench_chunkwise(model, tokens, args.steps, args.chunk_size)
    tps_recurrent = bench_recurrent(model, tokens, args.steps)

    print(f"parallel tokens/sec: {tps_parallel:.2f}")
    print(f"chunkwise tokens/sec: {tps_chunk:.2f}")
    print(f"recurrent tokens/sec: {tps_recurrent:.2f}")


if __name__ == "__main__":
    main()
