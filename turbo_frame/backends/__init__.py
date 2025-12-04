"""Backend registry helpers for TurboFrame."""
from __future__ import annotations

from typing import Iterable, List, Optional

from .base import BackendDetectionError, BackendRegistry, ComputeBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .metal import MetalBackend

_REGISTRY = BackendRegistry()
for backend_cls in (CUDABackend, MetalBackend, CPUBackend):
    _REGISTRY.register(backend_cls)


def available_backends() -> List[str]:
    """Return the human names of all currently available backends."""

    return [backend.name for backend in _REGISTRY.available()]


def select_backend(preferred: Optional[str | ComputeBackend] = None) -> ComputeBackend:
    """Return the best backend, optionally honoring a preferred name/class."""

    if preferred is None:
        return _REGISTRY.best()

    if isinstance(preferred, ComputeBackend):
        return preferred

    candidates = {backend.name.lower(): backend for backend in _REGISTRY.available()}
    key = preferred.lower()
    if key not in candidates:
        raise BackendDetectionError(
            f"Preferred backend '{preferred}' is not available. Found: {', '.join(candidates) or 'none'}"
        )
    return candidates[key]


__all__ = [
    "available_backends",
    "select_backend",
    "BackendDetectionError",
    "ComputeBackend",
]
