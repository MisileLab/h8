# RetNet MLX

A clean, reproducible MLX (Apple Silicon) implementation of RetNet-style
multi-scale retention with three computation paradigms:

- Parallel (training)
- Recurrent (O(1) decoding)
- Chunkwise recurrent (long-context)

This repo implements a pluggable DecayPolicy with learned static per-head
decay (default, monotonic heads) and a skeleton for conditional per-token
decay.

## Quickstart

```
pip install -e .
pytest -q
python scripts/train_char_lm.py --steps 500 --decay_policy learned_static
python scripts/generate.py --prompt "Hello" --max_new_tokens 200
```

## Modes

- parallel: full-sequence training, fast path for fixed or learned static decay
- recurrent: token-by-token decoding with cached state
- chunkwise: parallel within chunk, recurrent across chunks

## DecayPolicy

- fixed: predefined multi-scale decay
- learned_static: per-head decay, monotonic by construction (default)
- conditional: per-token per-head decay (skeleton, B2-ready)

## Repository Layout

- retnet_mlx/: core package
- scripts/: training, generation, benchmark
- tests/: pytest test suite
- data/: tiny char dataset

## Notes

Retention math uses float32 by default. Equivalence tests cover parallel,
recurrent, and chunkwise modes for small shapes.
