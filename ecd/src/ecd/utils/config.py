from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from omegaconf import OmegaConf


def load_config(
    paths: Sequence[str | Path], overrides: Sequence[str] | None = None
) -> Dict[str, Any]:
    configs = [OmegaConf.load(Path(path)) for path in paths]
    merged = OmegaConf.merge(*configs)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_cli(list(overrides)))
    return OmegaConf.to_container(merged, resolve=True)
