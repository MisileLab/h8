from __future__ import annotations

import gc
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TypeVar

import torch

T = TypeVar("T")


@dataclass
class VRAMInfo:
    """VRAM information for a device."""

    total: int  # Total memory in bytes
    free: int  # Free memory in bytes
    used: int  # Used memory in bytes
    reserved: int  # Reserved by PyTorch (CUDA only)

    @property
    def free_fraction(self) -> float:
        return self.free / self.total if self.total > 0 else 0.0

    @property
    def used_fraction(self) -> float:
        return self.used / self.total if self.total > 0 else 0.0


def get_vram_info(device: torch.device) -> VRAMInfo:
    """Get detailed VRAM information for a device."""
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        used = total - free
        return VRAMInfo(total=total, free=free, used=used, reserved=reserved)
    elif device.type == "mps":
        # MPS uses unified memory - estimate conservatively
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total = int(result.stdout.strip())
            # Conservative estimate: 50% of system memory available
            free = int(total * 0.5)
            return VRAMInfo(total=total, free=free, used=total - free, reserved=0)
        except Exception:
            total = 16 * 1024**3  # Assume 16GB
            return VRAMInfo(total=total, free=total // 2, used=total // 2, reserved=0)
    else:
        # CPU - use system memory
        total = 16 * 1024**3
        try:
            if os.name == "posix":
                with open("/proc/meminfo") as f:
                    mem_total = mem_available = 0
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_total = int(line.split()[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            mem_available = int(line.split()[1]) * 1024
                    if mem_total > 0:
                        return VRAMInfo(
                            total=mem_total,
                            free=mem_available,
                            used=mem_total - mem_available,
                            reserved=0,
                        )
        except Exception:
            pass
        return VRAMInfo(total=total, free=total // 2, used=total // 2, reserved=0)


def clear_memory_cache(device: torch.device) -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


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


@dataclass
class DynamicBatchConfig:
    """Configuration for dynamic batch sizing.
    
    Strategy: Start with a large batch size and reduce on OOM until stable.
    This finds the maximum usable batch size automatically.
    """

    enabled: bool = True
    initial_batch_size: int = 65536  # Start large, will reduce on OOM
    min_batch_size: int = 16  # Won't go below this
    oom_reduction_factor: float = 0.7  # Reduce batch by this factor on OOM
    max_oom_retries: int = 10  # Max retries before giving up


def estimate_training_batch_size(
    embedding_dim: int,
    output_dim: int,
    num_positives: int,
    num_negatives: int,
    device: torch.device,
    track_b_enabled: bool = False,
    track_b_k_pos: int = 50,
    track_b_m_neg: int = 1024,
    dtype: torch.dtype = torch.float32,
    amp_mode: str = "none",
    memory_fraction: float = 0.95,
    min_batch: int = 16,
    max_batch: int = 4096,
) -> int:
    """Estimate optimal training batch size based on available VRAM.

    This accounts for:
    - Teacher embeddings for anchors, positives, negatives
    - Student forward pass through all embeddings
    - Gradient storage for backpropagation
    - Track B additional memory if enabled

    Args:
        embedding_dim: Input/teacher embedding dimension
        output_dim: Student output dimension
        num_positives: Number of positive samples per anchor
        num_negatives: Number of negative samples per anchor
        device: Target device
        track_b_enabled: Whether Track B (listwise distillation) is enabled
        track_b_k_pos: Track B k_pos parameter
        track_b_m_neg: Track B m_neg parameter
        dtype: Data type for tensors
        amp_mode: AMP mode ('none', 'fp16', 'bf16')
        memory_fraction: Fraction of available VRAM to use
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Estimated optimal batch size
    """
    vram_info = get_vram_info(device)
    usable = int(vram_info.free * memory_fraction)

    dtype_size = 4  # float32
    if amp_mode in ("fp16", "bf16"):
        # Mixed precision reduces memory for activations but not weights/gradients
        dtype_size = 2.5  # Approximate average

    # Memory per batch sample breakdown:
    # 1. Teacher embeddings (input, stored)
    teacher_per_sample = embedding_dim * 4  # float32 always

    # 2. For each anchor we have:
    #    - 1 anchor embedding
    #    - num_positives positive embeddings
    #    - ~10 neighbor embeddings (for struct loss)
    #    - num_negatives negative embeddings
    samples_per_anchor = 1 + num_positives + 10 + num_negatives

    # 3. Student forward pass (input + hidden + output for each)
    #    Assume 2-layer MLP with hidden_dim ~= embedding_dim
    hidden_dim = max(embedding_dim, output_dim)
    student_activations = (embedding_dim + hidden_dim + output_dim) * dtype_size

    # 4. Gradients (roughly same as activations)
    gradient_memory = student_activations

    # 5. Optimizer states (Adam/AdamW: 2x model size for momentum + variance)
    #    This is amortized but let's add some overhead
    optimizer_overhead = output_dim * 0.5 * dtype_size

    # Total per anchor (not per sample)
    per_anchor = (
        teacher_per_sample * samples_per_anchor  # Teacher data
        + student_activations * samples_per_anchor  # Forward pass
        + gradient_memory * samples_per_anchor  # Gradients
        + optimizer_overhead  # Optimizer
    )

    # Track B adds significant memory
    if track_b_enabled:
        # Queue negatives: m_neg embeddings
        # Candidates: k_pos + m_neg embeddings per anchor
        # Additional forward pass for candidates
        track_b_candidates = track_b_k_pos + track_b_m_neg
        per_anchor += (
            embedding_dim * track_b_candidates * dtype_size  # Teacher candidates
            + output_dim * track_b_candidates * dtype_size  # Student candidates
            + gradient_memory * track_b_candidates  # Gradients
        )

    # Safety factor for fragmentation, intermediate tensors, etc.
    # Using lower factor since OOM retry will handle edge cases
    safety_factor = 1.5
    per_anchor *= safety_factor

    if per_anchor <= 0:
        return max_batch

    estimated = int(usable / per_anchor)

    # Round down to nearest power of 2 for efficiency
    if estimated > 0:
        import math

        estimated = 2 ** int(math.log2(estimated))

    result = max(min_batch, min(max_batch, estimated))
    return result


class OOMError(RuntimeError):
    """Raised when an out-of-memory error occurs."""

    pass


def is_oom_error(error: Exception) -> bool:
    """Check if an exception is an out-of-memory error."""
    error_str = str(error).lower()
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(error, RuntimeError):
        oom_indicators = [
            "out of memory",
            "cuda out of memory",
            "cudnn error",
            "cublas error",
            "failed to allocate",
            "not enough memory",
        ]
        return any(indicator in error_str for indicator in oom_indicators)
    return False


class DynamicBatchSizer:
    """Manages dynamic batch sizing with OOM retry logic.

    Strategy: Start with a large batch size and reduce on OOM until stable.
    This finds the maximum usable batch size automatically without gradual growth.
    """

    def __init__(
        self,
        config: DynamicBatchConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device

        # Start with large initial batch size
        self._current_batch_size = config.initial_batch_size
        self._oom_count = 0
        self._step = 0
        self._stable_batch_size: Optional[int] = None  # Set after first successful step

    @property
    def batch_size(self) -> int:
        """Current batch size."""
        return self._current_batch_size

    @property
    def oom_count(self) -> int:
        """Number of OOM errors encountered."""
        return self._oom_count

    @property
    def is_stabilized(self) -> bool:
        """Whether batch size has stabilized (no recent OOMs)."""
        return self._stable_batch_size is not None

    def on_step_success(self) -> None:
        """Called after a successful training step."""
        self._step += 1
        # Mark as stable after first success
        if self._stable_batch_size is None:
            self._stable_batch_size = self._current_batch_size
            vram_info = get_vram_info(self.device)
            print(
                f"Batch size stabilized at {self._current_batch_size} "
                f"(after {self._oom_count} OOM retries, "
                f"VRAM: {vram_info.used / 1024**3:.1f}GB used / {vram_info.total / 1024**3:.1f}GB total)"
            )

    def on_oom(self) -> int:
        """Called when OOM error occurs. Returns new batch size.

        Raises:
            OOMError: If max retries exceeded or batch size too small.
        """
        self._oom_count += 1
        self._stable_batch_size = None  # Reset stability

        if self._oom_count > self.config.max_oom_retries:
            raise OOMError(
                f"Max OOM retries ({self.config.max_oom_retries}) exceeded. "
                f"Last batch size was {self._current_batch_size}. "
                "Try using amp=bf16 or reducing model size."
            )

        # Reduce batch size
        old_size = self._current_batch_size
        new_size = int(old_size * self.config.oom_reduction_factor)

        # Round down to nearest power of 2 for GPU efficiency
        if new_size > 0:
            import math
            new_size = 2 ** int(math.log2(new_size))

        new_size = max(new_size, self.config.min_batch_size)

        if new_size < self.config.min_batch_size:
            raise OOMError(
                f"Batch size {new_size} below minimum {self.config.min_batch_size}. "
                "Try using amp=bf16 or reducing model size."
            )

        self._current_batch_size = new_size

        # Clear memory
        clear_memory_cache(self.device)

        vram_info = get_vram_info(self.device)
        print(
            f"OOM! Reducing batch: {old_size} -> {new_size} "
            f"(retry {self._oom_count}/{self.config.max_oom_retries}, "
            f"VRAM free: {vram_info.free / 1024**3:.1f}GB)"
        )

        return new_size

    def get_stats(self) -> dict:
        """Get current batch sizer statistics."""
        return {
            "current_batch_size": self._current_batch_size,
            "stable_batch_size": self._stable_batch_size,
            "oom_count": self._oom_count,
            "step": self._step,
            "is_stabilized": self.is_stabilized,
        }


def with_oom_retry(
    fn: Callable[..., T],
    batch_sizer: DynamicBatchSizer,
    *args,
    **kwargs,
) -> T:
    """Execute a function with automatic OOM retry.

    On OOM, clears cache, reduces batch size via batch_sizer, and retries.

    Args:
        fn: Function to execute
        batch_sizer: DynamicBatchSizer instance for batch management
        *args, **kwargs: Arguments to pass to fn

    Returns:
        Result of fn

    Raises:
        OOMError: If retries exhausted or batch too small
    """
    while True:
        try:
            result = fn(*args, **kwargs)
            batch_sizer.on_step_success()
            return result
        except Exception as e:
            if is_oom_error(e):
                batch_sizer.on_oom()
                # Retry with smaller batch
                continue
            else:
                raise
