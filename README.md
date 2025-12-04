# TurboFrame

TurboFrame is a friendly DataFrame facade that keeps the pandas API you already know while quietly choosing the fastest compute backend on your machine. Stay in pure Python, write tidy pandas code, and let TurboFrame decide whether CUDA, Apple Metal, or a regular CPU run should own the work.

## Why TurboFrame exists

Modern laptops and workstations ship with powerful GPUs, yet most data manipulation code still lives on the CPU because moving to GPU-centric libraries usually means rewriting every call site. TurboFrame was built to remove that friction:

1. **API compatibility first.** The methods exposed on `TurboFrame` and `TurboSeries` mirror the pandas experience (select, filter, groupby, merge, etc.).
2. **Lazy graphs behind the scenes.** Each transformation is queued rather than executed immediately. When you finally request the result, TurboFrame optimizes and dispatches the graph in one go.
3. **Automatic backend selection.** A registry probes CUDA/cuDF, Apple Metal/MLX, and pandas, picking the highest priority backend that is available at runtime. You can override the choice whenever you like.

## Installation

```bash
git clone https://github.com/M1tsumi/turboframe.git
cd turboframe
pip install -r requirements.txt  # pandas + optional cudf/mlx/cupy per platform
```

TurboFrame itself is pure Python and depends only on pandas. GPU backends are optional extras; install `cudf`/`cupy` for NVIDIA cards or `mlx` for Apple Silicon when you want accelerated paths.

## Quick start

```python
import pandas as pd
from turbo_frame import TurboFrame

pdf = pd.read_csv("sales.csv")
tf = TurboFrame(pdf)

result = (
    tf.with_columns({"net": tf["revenue"] - tf["cost"]})
      .groupby_agg(by=["region"], agg={"net": "sum"})
      .sort_values("net", ascending=False)
      .head(10)
      .compute()  # materializes using the best backend
)

print(result)
```

Because operations are lazy, you can branch and reuse the same base frame without incurring extra compute until you call `compute()` or `to_pandas()`.

## Core features in v1.0.0

### Lazy orchestration
- Every `TurboFrame` / `TurboSeries` method appends an `Operation` descriptor instead of executing immediately.
- Computation graphs are replayed once, reducing redundant intermediate allocations.

### Backend abstraction layer
- `ComputeBackend` defines `to_native`, `execute`, and `to_pandas` hooks.
- Included backends:
  - `CUDABackend` (cuDF + CuPy) for NVIDIA GPUs.
  - `MetalBackend` (MLX) for Apple Silicon.
  - `CPUBackend` (pandas) as the universal fallback.
- `select_backend()` lets you override automatic detection per frame or per compute call.

### GPU-friendly operations
TurboFrame queues the pandas-compatible callables listed below, which map neatly to GPU primitives:

- Aggregations: `aggregate`, `groupby_agg`, `count`, `mean`, etc.
- Transformations: `assign`, `with_columns`, `apply`, `map_column`, `astype`, `fillna`, `replace`.
- Filtering: `filter`, `query`, `dropna`, `where` (Series), boolean indexing.
- Sorting: `sort_values`, `sort_index`.
- Joins & unions: `merge`, `concat`.
- String helpers on Series: `str_lower`, `str_upper`, `str_contains`.

### Memory-aware caching
- Results are cached per backend key so repeated `.compute()` calls avoid recomputation.
- GPU materialization requests reuse the same backend when chaining frames together (e.g., during `merge` or `concat`).

## Working with backends

```python
from turbo_frame import TurboFrame, select_backend

tf = TurboFrame(pdf)
tf_cuda = tf.with_backend("cuda")  # force a backend by name

print(TurboFrame.available_backends())  # e.g. ["CUDA", "CPU"]

backend = select_backend()  # returns the best backend object
tf.compute(backend="cpu")  # override at call time
```

Backends declare their availability dynamically, so shipping TurboFrame to colleagues without GPUs is safe—everything falls back to pandas automatically.

## Project layout

```
turbo_frame/
├── __init__.py      # public API surface
├── _lazy.py         # shared lazy-object plumbing
├── operations.py    # dataclasses describing queued operations
├── frame.py         # TurboFrame facade
├── series.py        # TurboSeries facade
└── backends/
    ├── __init__.py
    ├── base.py
    ├── cpu.py
    ├── cuda.py
    └── metal.py
```

## Roadmap

- Add a disk spill manager for GPUs with limited VRAM.
- Expand the string accessor coverage to match the pandas `.str` namespace.
- Ship benchmark notebooks contrasting pandas, cuDF, and MLX backends on common workloads.

## License

TurboFrame is released under the Apache 2.0 License (see `LICENSE`).
