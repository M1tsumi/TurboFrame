"""CUDA backend using cuDF/CuPy when available."""
from __future__ import annotations

from typing import Any

import pandas as pd

try:  # pragma: no cover - optional dependency
    import cudf  # type: ignore
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - fallback
    cudf = None  # type: ignore
    cp = None  # type: ignore

from .base import ComputeBackend


class CUDABackend(ComputeBackend):
    name = "CUDA"
    priority = 30
    backend_key = "cudf"

    @classmethod
    def is_available(cls) -> bool:
        return cudf is not None and cp is not None

    def to_native(self, obj: Any) -> Any:
        if cudf is None:
            raise RuntimeError("cuDF is not available")
        if isinstance(obj, cudf.DataFrame) or isinstance(obj, cudf.Series):
            return obj
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return cudf.from_pandas(obj)
        return cudf.from_pandas(pd.DataFrame(obj))

    def to_pandas(self, obj: Any) -> Any:
        if cudf is None:
            raise RuntimeError("cuDF is not available")
        return obj.to_pandas()


__all__ = ["CUDABackend"]
