"""Backend abstraction for TurboFrame."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from ..operations import Operation


class ComputeBackend(ABC):
    """Abstract backend capable of executing lazy operations."""

    name: str = "base"
    priority: int = 0
    backend_key: str = "pandas"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this backend can be used on the current machine."""

    @abstractmethod
    def to_native(self, obj: Any) -> Any:
        """Convert a pandas object into the backend-native representation."""

    @abstractmethod
    def to_pandas(self, obj: Any) -> Any:
        """Convert a backend-native object back to pandas."""

    def execute(self, base_obj: Any, operations: Iterable[Operation]) -> Any:
        """Apply the lazily recorded operations using this backend."""

        native = self.to_native(base_obj)
        for op in operations:
            native = self._apply_operation(op, native)
        return self.to_pandas(native)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_operation(self, operation: Operation, native: Any) -> Any:
        callable_ = self._select_callable(operation)
        if callable_ is None:
            raise NotImplementedError(
                f"Operation '{operation.description}' not supported by backend {self.name}."
            )
        return callable_(native, self)

    def _select_callable(self, operation: Operation):
        match self.backend_key:
            case "cudf":
                return operation.cudf_callable or operation.pandas_callable
            case "metal":
                return operation.metal_callable or operation.pandas_callable
            case _:
                return operation.pandas_callable


class BackendDetectionError(RuntimeError):
    """Raised when no backend can be selected."""


class BackendRegistry:
    """Simple registry keeping backend classes sorted by priority."""

    def __init__(self) -> None:
        self._backend_classes: list[type[ComputeBackend]] = []

    def register(self, backend_cls: type[ComputeBackend]) -> None:
        if backend_cls not in self._backend_classes:
            self._backend_classes.append(backend_cls)
            self._backend_classes.sort(key=lambda cls: cls.priority, reverse=True)

    def available(self) -> list[ComputeBackend]:
        return [cls() for cls in self._backend_classes if cls.is_available()]

    def best(self) -> ComputeBackend:
        available = self.available()
        if not available:
            raise BackendDetectionError("No compute backend available.")
        return available[0]
