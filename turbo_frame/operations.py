"""Lazy operation descriptors shared by frames and series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    from .backends.base import ComputeBackend

NativeCallable = Callable[[Any, "ComputeBackend"], Any]


@dataclass(frozen=True)
class Operation:
    """Describe a lazy operation applied to DataFrame or Series objects."""

    description: str
    pandas_callable: NativeCallable
    cudf_callable: Optional[NativeCallable] = None
    metal_callable: Optional[NativeCallable] = None


OperationList = list[Operation]
