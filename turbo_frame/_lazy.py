"""Common lazy-evaluation utilities shared by TurboFrame and TurboSeries."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, TypeVar

from .backends import ComputeBackend, available_backends, select_backend
from .operations import Operation

_SENTINEL = object()
T_Lazy = TypeVar("T_Lazy", bound="LazyObject")


class LazyObject:
    """Base class that stores the seed data and queued operations."""

    def __init__(
        self,
        base: Any,
        *,
        operations: Optional[Iterable[Operation]] = None,
        preferred_backend: str | ComputeBackend | None = None,
    ) -> None:
        self._base = base
        self._operations: List[Operation] = list(operations or [])
        self._preferred_backend: str | ComputeBackend | None = preferred_backend
        self._cache_backend_key: str | None = None
        self._cache_value: Any | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def with_backend(self, name: str | ComputeBackend | None) -> "LazyObject":
        """Return a copy that prefers a specific backend when computing."""

        return self._spawn(self.__class__, preferred_backend=name)

    def describe_plan(self) -> list[str]:
        """Return human readable descriptions of queued operations."""

        return [op.description for op in self._operations]

    def has_pending_operations(self) -> bool:
        return bool(self._operations)

    @staticmethod
    def available_backends() -> list[str]:
        return available_backends()

    def compute(self, backend: str | ComputeBackend | None = None) -> Any:
        """Materialize the object using the requested or preferred backend."""

        backend_obj = self._resolve_backend(backend)
        cached = self._cached_value_for_backend(backend_obj)
        if cached is not None:
            return cached
        result = self._compute_with_backend(backend_obj)
        self._store_cache(backend_obj, result)
        return result

    def to_pandas(self, backend: str | ComputeBackend | None = None) -> Any:
        """Alias for compute() to match pandas naming."""

        return self.compute(backend=backend)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_backend(self, backend: str | ComputeBackend | None) -> ComputeBackend:
        candidate = backend if backend is not None else self._preferred_backend
        if isinstance(candidate, ComputeBackend):
            return candidate
        return select_backend(candidate)

    def _compute_with_backend(self, backend: ComputeBackend) -> Any:
        return backend.execute(self._base, self._operations)

    def _materialize_with_backend(self, backend: ComputeBackend) -> Any:
        cached = self._cached_value_for_backend(backend)
        if cached is not None:
            return cached
        result = self._compute_with_backend(backend)
        self._store_cache(backend, result)
        return result

    def _spawn(
        self,
        cls: type[T_Lazy],
        *,
        operation: Operation | None = None,
        base: Any | None = None,
        operations: Optional[Iterable[Operation]] = None,
        preferred_backend: str | ComputeBackend | None | object = _SENTINEL,
    ) -> T_Lazy:
        new_obj = object.__new__(cls)
        new_obj._base = self._base if base is None else base
        if operations is not None:
            new_obj._operations = list(operations)
        else:
            new_obj._operations = list(self._operations)
        if operation is not None:
            new_obj._operations.append(operation)
        if preferred_backend is _SENTINEL:
            new_obj._preferred_backend = self._preferred_backend
        else:
            new_obj._preferred_backend = preferred_backend
        new_obj._cache_backend_key = None
        new_obj._cache_value = None
        return new_obj

    def _copy_operations(self) -> list[Operation]:
        return list(self._operations)

    # ------------------------------------------------------------------
    # Operation helpers
    # ------------------------------------------------------------------
    def _queue_operation(
        self,
        *,
        description: str,
        pandas_callable,
        cudf_callable=None,
        metal_callable=None,
        result_cls: type[T_Lazy] | None = None,
    ) -> T_Lazy:
        op = Operation(
            description=description,
            pandas_callable=pandas_callable,
            cudf_callable=cudf_callable,
            metal_callable=metal_callable,
        )
        target_cls = result_cls or self.__class__
        return self._spawn(target_cls, operation=op)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _store_cache(self, backend: ComputeBackend, value: Any) -> None:
        self._cache_backend_key = backend.backend_key
        self._cache_value = value

    def _cached_value_for_backend(self, backend: ComputeBackend) -> Any | None:
        if self._cache_backend_key == backend.backend_key and self._cache_value is not None:
            try:
                return self._cache_value.copy(deep=False)
            except AttributeError:
                return self._cache_value
        return None
