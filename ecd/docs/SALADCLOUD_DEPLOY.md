# SaladCloud Deployment Guide

This guide covers deploying the ECD (Embedding Compression DB) project to [SaladCloud](https://salad.com) for running Track-B experiments on distributed GPU infrastructure.

## Overview

SaladCloud uses a container-based deployment model. You'll need to:
1. Build a Docker image with your code
2. Push it to a container registry
3. Create a container group via API or web console
4. Monitor and retrieve results

## Quick Start

### 1. Build the Docker Image

```bash
# Build the SaladCloud-optimized image
docker build -f Dockerfile.saladcloud -t ecd-saladcloud:latest .

# Tag for your registry
docker tag ecd-saladcloud:latest YOUR_REGISTRY/ecd-saladcloud:latest

# Push to registry
docker push YOUR_REGISTRY/ecd-saladcloud:latest
```

### 2. Deploy via Python Script

```bash
# Set credentials
export SALAD_API_KEY="your_api_key"
export SALAD_ORG_NAME="your_organization"

# Create and start training
python scripts/saladcloud_launch.py \
    --create \
    --image YOUR_REGISTRY/ecd-saladcloud:latest \
    --gpu rtx4090 \
    --steps 20000
```

### 3. Monitor Progress

```bash
# Check status
python scripts/saladcloud_launch.py --status --group-name ecd-track-b-XXXXX

# Get logs
python scripts/saladcloud_launch.py --logs --group-name ecd-track-b-XXXXX

# Monitor until completion
python scripts/saladcloud_launch.py --monitor --group-name ecd-track-b-XXXXX
```

## Deployment Scripts

### `Dockerfile.saladcloud`

Optimized Docker image for SaladCloud:
- Based on PyTorch with CUDA 12.1
- Pre-installs dependencies using uv
- Includes health check endpoint
- Auto-runs training on container start

### `scripts/saladcloud_deploy.sh`

Main script that runs inside the container:
- Sets up environment
- Prepares data
- Runs Track-B experiments
- Uploads results to cloud storage

### `scripts/saladcloud_launch.py`

Python API client for SaladCloud:
- Create/start container groups
- Monitor job status
- Retrieve logs
- Stop/delete containers

## Configuration

### Environment Variables

Set these in SaladCloud container configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACK_B_STEPS` | `20000` | Training steps |
| `TRACK_B_SEEDS` | `0,1,2` | Seeds (comma-separated) |
| `QUICK_MODE` | `false` | Run quick test (10 steps) |
| `UPLOAD_RESULTS` | `false` | Upload to cloud storage |
| `RESULTS_BUCKET` | `""` | S3/GCS/Azure bucket URL |

### GPU Classes

Available GPU classes for `--gpu` option:

| Option | SaladCloud Classes |
|--------|-------------------|
| `rtx4090` | RTX 4090 24GB |
| `rtx3090` | RTX 3090 24GB |
| `a100` | A100 40GB/80GB |
| `any_24gb` | Any 24GB+ GPU |
| `any` | Any available GPU |

### Resource Requirements

Recommended minimums:
- **CPU**: 4 cores
- **Memory**: 32GB
- **GPU VRAM**: 16GB+ (24GB recommended)
- **Disk**: 50GB

## Using the Web Console

If you prefer the SaladCloud web console:

1. Go to [SaladCloud Console](https://portal.salad.com)
2. Create a new Container Group
3. Configure:
   - **Image**: `YOUR_REGISTRY/ecd-saladcloud:latest`
   - **GPU**: RTX 4090 or equivalent
   - **CPU**: 4 cores
   - **Memory**: 32768 MB
   - **Environment Variables**: Add the variables from the table above
   - **Port**: 8080 (for health checks)
4. Deploy and monitor

## Results Upload

### AWS S3

```bash
# Set in container environment
RESULTS_BUCKET=s3://your-bucket/ecd-results
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
UPLOAD_RESULTS=true
```

### Google Cloud Storage

```bash
RESULTS_BUCKET=gs://your-bucket/ecd-results
# Mount service account key or use workload identity
UPLOAD_RESULTS=true
```

### Azure Blob Storage

```bash
RESULTS_BUCKET=https://youraccount.blob.core.windows.net/container
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
UPLOAD_RESULTS=true
```

## API Reference

### List Container Groups

```python
from scripts.saladcloud_launch import SaladCloudClient

client = SaladCloudClient(api_key, org_name)
groups = client.list_container_groups("ecd-experiments")
```

### Create Container Group

```python
from scripts.saladcloud_launch import create_ecd_container_group

result = create_ecd_container_group(
    client=client,
    project_name="ecd-experiments",
    image="your-registry/ecd:latest",
    steps=20000,
    seeds="0,1,2",
    gpu_class="rtx4090"
)
```

### Monitor Status

```python
from scripts.saladcloud_launch import monitor_container_group

success = monitor_container_group(
    client=client,
    project_name="ecd-experiments",
    group_name="ecd-track-b-123",
    timeout=7200  # 2 hours
)
```

## Troubleshooting

### Container Won't Start

1. Check image is accessible from SaladCloud
2. Verify GPU class availability
3. Check resource requirements aren't too high

```bash
# Check container group status
python scripts/saladcloud_launch.py --status --group-name YOUR_GROUP
```

### Training Fails

1. Check container logs:
```bash
python scripts/saladcloud_launch.py --logs --group-name YOUR_GROUP
```

2. Common issues:
   - CUDA out of memory: Reduce batch size or queue size
   - Data loading errors: Check network connectivity
   - Permission errors: Verify cloud storage credentials

### Results Not Uploading

1. Verify `UPLOAD_RESULTS=true`
2. Check `RESULTS_BUCKET` is set correctly
3. Verify cloud credentials are mounted/set

## Cost Optimization

1. **Use spot instances** when available
2. **Start with `--quick`** to verify setup
3. **Set appropriate timeouts** to avoid runaway costs
4. **Delete container groups** immediately after completion

```bash
# Quick test first
python scripts/saladcloud_launch.py --create --image IMG --quick

# Then full run
python scripts/saladcloud_launch.py --create --image IMG --steps 50000

# Clean up when done
python scripts/saladcloud_launch.py --delete --group-name GROUP_NAME
```

## Example Workflows

### Full Training Run

```bash
# 1. Build and push image
docker build -f Dockerfile.saladcloud -t myregistry/ecd:v1 .
docker push myregistry/ecd:v1

# 2. Set credentials
export SALAD_API_KEY="..."
export SALAD_ORG_NAME="..."

# 3. Launch with monitoring
python scripts/saladcloud_launch.py \
    --create \
    --image myregistry/ecd:v1 \
    --steps 50000 \
    --seeds "0,1,2" \
    --gpu rtx4090 \
    --results-bucket s3://my-bucket/results \
    --monitor

# 4. Results will be in S3 when complete
```

### Quick Validation

```bash
# Test with minimal training
python scripts/saladcloud_launch.py \
    --create \
    --image myregistry/ecd:v1 \
    --quick \
    --dry-run  # See what would happen first

# Then actually run
python scripts/saladcloud_launch.py \
    --create \
    --image myregistry/ecd:v1 \
    --quick
```

### Batch Experiments

```bash
# Run multiple configurations
for tau in 0.05 0.07 0.10; do
    python scripts/saladcloud_launch.py \
        --create \
        --image myregistry/ecd:v1 \
        --steps 20000 \
        # Note: would need to add tau as env var in script
done
```
