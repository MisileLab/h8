from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def now_ms() -> float:
    return time.time() * 1000.0


@contextmanager
def timer() -> Iterator[float]:
    start = time.time()
    yield start
