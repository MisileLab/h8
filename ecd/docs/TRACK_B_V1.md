# Track-B v1: Listwise Distillation Implementation

## Overview

Track-B v1 implements listwise distillation using soft targets from the teacher model, enhanced with:
- **Projection head**: Two-layer MLP (D→512→128) with optional skip connection
- **Candidate set**: K teacher top-K positives + M memory queue negatives
- **Cosine similarity**: All vectors are L2-normalized before scoring
- **Listwise KL loss**: Student mimics teacher's ranking distribution over candidates
- **False negative filtering**: Remove false negatives from candidate set

## Architecture

### Student Model (Track-B Projection Head)

```
Teacher embedding (D)
    ↓
Linear(D→512) → GELU → Dropout
    ↓
Linear(512→128)
    ↓
[Optional] Skip connection: α * Linear(D→128)
    ↓
L2 Normalization → Output (128-dim)
```

**Key parameters:**
- `hidden_dim`: 512 (fixed in Track-B)
- `out_dim`: 128 (fixed in Track-B)
- `track_b_use_skip`: Enable skip connection (default: true)
- `track_b_alpha_init`: Learnable skip weight α (default: 0.1)

### Candidate Set Construction

For each batch anchor:
1. **Positives**: Teacher top-K neighbors (K=50 default)
   - Retrieved from precomputed teacher top-k indices
2. **Negatives**: M samples from memory queue (M=1024 default)
   - Queue stores teacher document embeddings (FIFO)
   - Queue size: 32k default (64k optional)

### False Negative Filtering

Two modes to remove false negatives from candidate set:

**Mode 1: threshold (default)**
- Remove negatives where `cos(anchor, negative) > threshold`
- Default threshold: 0.8

**Mode 2: top_percent**
- Remove top x% of negatives by teacher score
- Default: 2% (0.02)

**Mode 3: none**
- No filtering

### Loss Function

```
teacher_scores = cos(t_q, candidates) / τ
student_scores = cos(s_q, s_candidates) / τ

p_T = softmax(teacher_scores)
p_S = softmax(student_scores)

loss_listwise = KL(p_T || p_S) = sum p_T * (log p_T - log p_S)
```

**Parameters:**
- `tau` (τ): Temperature for softmax (default: 0.07)

### Mixed Loss Training

Total loss is a weighted combination of Track-A and Track-B:

```
loss = (1 - λ) * loss_track_a + λ * loss_listwise
```

- `λ` (lambda): Mix coefficient (default: 1.0)
- λ=1.0: Track-B only
- λ=0.5: Equal mix
- λ=0.0: Track-A only

## Environment Variable Controls

### Primary Controls

| Variable | Default | Description |
|----------|----------|-------------|
| `TRACK_B_ENABLE` | `false` | Enable Track-B v1 training |
| `TRACK_B_TAU` | `0.07` | Temperature for softmax in listwise loss |
| `TRACK_B_K` | `50` | Number of teacher top-K positives per anchor |
| `TRACK_B_M` | `1024` | Number of memory queue negatives per anchor |
| `TRACK_B_QUEUE_SIZE` | `32000` | FIFO queue size for teacher embeddings |

### False Negative Filtering

| Variable | Default | Description |
|----------|----------|-------------|
| `TRACK_B_FALSE_NEG_FILTER_MODE` | `threshold` | Filter mode: `none`, `threshold`, `top_percent` |
| `TRACK_B_FALSE_NEG_THRESHOLD` | `0.8` | Cosine threshold for false negatives (mode=threshold) |
| `TRACK_B_FALSE_NEG_TOP_PERCENT` | `0.02` | Top percent to remove (mode=top_percent) |

### Loss Mixing

| Variable | Default | Description |
|----------|----------|-------------|
| `TRACK_B_MIX_LAMBDA` | `1.0` | Mix coefficient λ for Track-B vs Track-A (0.0 to 1.0) |

### Queue & Model

| Variable | Default | Description |
|----------|----------|-------------|
| `TRACK_B_QUEUE_CPU_FALLBACK` | `true` | Use CPU storage if GPU memory insufficient |
| `TRACK_B_USE_SKIP` | `true` | Enable skip connection in projection head |
| `TRACK_B_ALPHA_INIT` | `0.1` | Learnable skip weight α initialization |

## Usage Examples

### 1. Basic Track-B v1 Single Run

```bash
TRACK_B_ENABLE=true \
TRACK_B_K=50 \
TRACK_B_M=1024 \
TRACK_B_TAU=0.07 \
bash scripts/run_track_a.sh single
```

### 2. Track-B with Custom Hyperparameters

```bash
TRACK_B_ENABLE=true \
TRACK_B_K=100 \
TRACK_B_M=2048 \
TRACK_B_TAU=0.05 \
TRACK_B_QUEUE_SIZE=64000 \
TRACK_B_FALSE_NEG_FILTER_MODE=top_percent \
TRACK_B_FALSE_NEG_TOP_PERCENT=0.01 \
TRACK_B_MIX_LAMBDA=0.5 \
bash scripts/run_track_a.sh single
```

### 3. Track-B Sweep Over Hyperparameters

```bash
TRACK_B_ENABLE=true \
TRACK_B_K="[25,50,100]" \
TRACK_B_M="[512,1024,2048]" \
TRACK_B_TAU="[0.05,0.07,0.10]" \
bash scripts/run_track_a.sh sweep
```

### 4. Mixed Training (Track-A + Track-B)

```bash
TRACK_B_ENABLE=true \
TRACK_B_MIX_LAMBDA=0.5 \
bash scripts/run_track_a.sh single
```

### 5. Smoke Test (Dry Run)

```bash
bash scripts/test_trackb_smoke.sh
```

## Configuration Files

### Model Config: `configs/model/track_b.yaml`

```yaml
type: "track_b"
in_dim: null  # Auto-detected from teacher embeddings
out_dim: 128
hidden_dim: 512
dropout: 0.0
normalize: true
track_b_use_skip: true
track_b_alpha_init: 0.1
```

### Training Config: `configs/train/default.yaml`

Track-B section:

```yaml
track_b:
  enable: false
  k_pos: 50
  m_neg: 1024
  tau: 0.07
  queue_size: 32000
  queue_cpu_fallback: true
  false_neg_filter:
    mode: "threshold"
    threshold: 0.8
    top_percent: 0.02
  mix:
    lambda: 1.0
```

### Sweep Configs

- `configs/sweep/trackB_grid.yaml`: Track-B-specific grid search parameters
- `configs/sweep/quick_hyper.yaml`: Updated with Track-B knobs

## Implementation Details

### Memory Queue

- **FIFO (First-In-First-Out)**: New embeddings replace oldest when full
- **CPU fallback**: Automatically use CPU storage if GPU memory is low
- **Initialization**: Queue is warmed up with random samples before training starts
- **Storage**: Teacher document embeddings (not student embeddings)

### Training Loop Integration

1. Warmup: Fill queue with random teacher embeddings (if empty)
2. For each batch:
   - Sample anchors and get teacher top-K
   - Enqueue current batch teacher embeddings
   - Sample M negatives from queue
   - Build candidates (K positives + filtered M negatives)
   - Compute teacher scores: `cos(t_anchor, candidates)`
   - Compute student scores: `cos(s_anchor, s_candidates)`
   - Compute listwise KL loss
   - Mix with Track-A losses
   - Backpropagate

### Logging

Additional metrics logged when Track-B is enabled:

| Metric | Description |
|---------|-------------|
| `track_b_enable` | Whether Track-B is enabled |
| `track_b_tau` | Temperature τ |
| `track_b_k` | Number of positives K |
| `track_b_m` | Number of negatives M |
| `track_b_queue_size` | Queue size |
| `track_b_false_neg_filter_mode` | Filter mode |
| `track_b_false_neg_threshold` | Threshold value |
| `track_b_false_neg_top_percent` | Top percent value |
| `track_b_mix_lambda` | Mix coefficient λ |
| `track_b_p_t_entropy` | Average entropy of teacher distribution |
| `track_b_filtered_neg_ratio` | Fraction of negatives filtered |
| `loss_listwise` | Listwise KL loss value |

## Verification

Run smoke test:

```bash
# Dry run (checks configuration)
bash scripts/test_trackb_smoke.sh

# Actual run (requires cached data)
TRACK_B_ENABLE=true STEPS=200 bash scripts/run_track_a.sh single
```

Run unit tests:

```bash
python -m pytest tests/test_track_b.py -v
```

## Comparison: Track-A vs Track-B

| Feature | Track-A | Track-B |
|----------|----------|----------|
| Training objective | Multi-task (distill + rank + struct) | Listwise KL distillation |
| Negatives | Hard negatives from teacher tail | Memory queue + false negative filtering |
| Student architecture | MLP (D→hidden→128) | Two-layer MLP with skip (D→512→128) |
| Scoring | Cosine similarity | Cosine similarity (same) |
| Loss | InfoNCE + MSE + distillation | KL divergence on ranking distributions |

## Notes

- Track-B v1 is **feature-flagged**: Set `TRACK_B_ENABLE=true` to activate
- When Track-B is enabled, `model.type` is automatically set to `"track_b"`
- Track-A functionality is **preserved**: Can mix both losses or use Track-B only
- Memory queue uses **teacher embeddings**: Student learns to mimic teacher's ranking
- False negative filtering is **optional**: Set mode to `"none"` to disable
