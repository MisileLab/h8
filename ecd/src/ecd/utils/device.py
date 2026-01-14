from __future__ import annotations

import os
from typing import Tuple

import torch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device(device)


def get_available_memory(device: torch.device) -> int:
    """Get available memory in bytes for the given device.

    For CUDA: uses torch.cuda.mem_get_info()
    For MPS: estimates from system memory (conservative 50% of total)
    For CPU: uses system memory
    """
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return free
    elif device.type == "mps":
        # MPS doesn't expose memory info directly
        # Use system memory as proxy (Apple unified memory)
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total_mem = int(result.stdout.strip())
            # Conservative: assume 50% available for MPS operations
            return int(total_mem * 0.5)
        except Exception:
            # Fallback: assume 8GB available
            return 8 * 1024**3
    else:
        # CPU: use system memory
        try:
            import subprocess

            if os.name == "posix":
                if os.uname().sysname == "Darwin":
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    return int(int(result.stdout.strip()) * 0.7)
                else:
                    # Linux
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                return int(line.split()[1]) * 1024
        except Exception:
            pass
        # Fallback: assume 8GB
        return 8 * 1024**3


def estimate_batch_size(
    num_items: int,
    embedding_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    memory_fraction: float = 0.7,
    min_batch: int = 64,
    max_batch: int = 8192,
) -> int:
    """Estimate optimal batch size for matrix operations based on available memory.

    For computing similarity: batch @ matrix.T where matrix is (num_items, embedding_dim)
    Memory needed per batch row: num_items * sizeof(dtype) for the result row
    Plus the batch itself: embedding_dim * sizeof(dtype)

    Args:
        num_items: Total number of items (columns in result matrix)
        embedding_dim: Dimension of embeddings
        device: Target device
        dtype: Data type for tensors
        memory_fraction: Fraction of available memory to use
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Estimated optimal batch size
    """
    available = get_available_memory(device)
    usable = int(available * memory_fraction)

    # Memory per batch element:
    # - Input batch row: embedding_dim * dtype_size
    # - Output similarity row: num_items * dtype_size
    # - Some overhead for intermediate tensors (2x safety factor)
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    memory_per_row = (embedding_dim + num_items) * dtype_size * 2

    if memory_per_row <= 0:
        return max_batch

    estimated = usable // memory_per_row
    return max(min_batch, min(max_batch, estimated))


def estimate_encode_batch_size(
    embedding_dim: int,
    output_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    memory_fraction: float = 0.5,
    min_batch: int = 64,
    max_batch: int = 4096,
) -> int:
    """Estimate optimal batch size for model encoding.

    Args:
        embedding_dim: Input embedding dimension
        output_dim: Output embedding dimension
        device: Target device
        dtype: Data type for tensors
        memory_fraction: Fraction of available memory to use
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Estimated optimal batch size
    """
    available = get_available_memory(device)
    usable = int(available * memory_fraction)

    dtype_size = torch.tensor([], dtype=dtype).element_size()
    # Memory per sample: input + output + intermediate (4x safety for MLP layers)
    memory_per_sample = (embedding_dim + output_dim) * dtype_size * 4

    if memory_per_sample <= 0:
        return max_batch

    estimated = usable // memory_per_sample
    return max(min_batch, min(max_batch, estimated))
