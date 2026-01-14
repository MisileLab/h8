from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    base_seed = np.random.get_state()[1][0]
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def get_torch_generator(seed: Optional[int]) -> Optional[object]:
    if torch is None or seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
