# Changelog

All notable changes to TurboFrame will be documented in this file.

## [1.0.0] - 2025-12-04
### Added
- **TurboFrame core** – pandas-compatible DataFrame facade with lazy evaluation, backend selection, and comprehensive column/groupby/join helpers.
- **TurboSeries companion** – column-level wrapper supporting mapping, filtering, string ops, and statistic helpers to mirror Series ergonomics.
- **Lazy operation graph** – reusable `Operation` descriptors, per-backend caching, and shared `_lazy` utilities for both frames and series.
- **Backend abstraction layer** – `ComputeBackend` base class, registry utilities, and three concrete backends (CUDA via cuDF/CuPy, Metal via MLX, CPU via pandas fallback).
- **Documentation** – refreshed README with motivation, installation, feature list, backend guidance, roadmap, and project layout.
- **Tooling** – `.gitignore` tuned for Python builds plus `plan.md` exclusion so internal plans stay private.
