"""CPU (pandas) backend implementation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import ComputeBackend


class CPUBackend(ComputeBackend):
    name = "CPU"
    priority = 10
    backend_key = "pandas"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def to_native(self, obj: Any) -> Any:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.copy(deep=False)
        return pd.DataFrame(obj)

    def to_pandas(self, obj: Any) -> Any:
        return obj


__all__ = ["CPUBackend"]
