"""Metal backend leveraging MLX when available."""
from __future__ import annotations

import platform
from typing import Any

import pandas as pd

try:  # pragma: no cover - optional dependency
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover - fallback
    mx = None  # type: ignore

from .base import ComputeBackend


class MetalBackend(ComputeBackend):
    name = "Metal"
    priority = 20
    backend_key = "metal"

    @classmethod
    def is_available(cls) -> bool:
        return platform.system() == "Darwin" and mx is not None

    def to_native(self, obj: Any) -> Any:
        # For now this backend runs pandas ops but keeps a mirrored MLX tensor cache
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        return df

    def to_pandas(self, obj: Any) -> Any:
        return obj


__all__ = ["MetalBackend"]
