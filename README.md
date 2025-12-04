# TurboFrame

TurboFrame is a DataFrame wrapper that keeps the pandas API you know while picking the fastest compute backend available on your machine. Write normal pandas code and let TurboFrame figure out whether to run on CUDA, Apple Metal, or plain CPU.

## Why TurboFrame exists

GPUs are everywhere now, but most data code still runs on the CPU because switching to GPU libraries means rewriting everything. TurboFrame fixes that:

1. **Same API you already use.** `TurboFrame` and `TurboSeries` work like pandas—select, filter, groupby, merge, all the usual stuff.
2. **Lazy execution.** Operations queue up instead of running immediately. When you ask for results, TurboFrame runs the whole graph at once.
3. **Picks the best backend automatically.** It checks for CUDA/cuDF, Apple Metal/MLX, and falls back to pandas. Override whenever you want.

## Installation

```bash
git clone https://github.com/M1tsumi/turboframe.git
cd turboframe
pip install -r requirements.txt  # pandas + optional cudf/mlx/cupy per platform
```

TurboFrame itself is pure Python and only needs pandas. For GPU acceleration, install `cudf`/`cupy` on NVIDIA or `mlx` on Apple Silicon.

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

Since operations are lazy, you can branch off the same base frame multiple times without extra compute until you call `compute()` or `to_pandas()`.

## Core features in v1.0.0

### Lazy execution
`TurboFrame` and `TurboSeries` methods just append an `Operation` descriptor. The graph runs once when you materialize, so you don't waste time on intermediate results.

### Backend abstraction
`ComputeBackend` provides `to_native`, `execute`, and `to_pandas` hooks. Included:
- `CUDABackend` — cuDF + CuPy for NVIDIA GPUs
- `MetalBackend` — MLX for Apple Silicon
- `CPUBackend` — pandas fallback

Use `select_backend()` to override detection per frame or per compute call.

### Supported operations
Aggregations (`aggregate`, `groupby_agg`, `count`, `mean`, etc.), transformations (`assign`, `with_columns`, `apply`, `fillna`, `replace`), filtering (`filter`, `query`, `dropna`, boolean indexing), sorting (`sort_values`, `sort_index`), joins (`merge`, `concat`), and basic string methods on Series (`str_lower`, `str_upper`, `str_contains`).

### Caching
Results cache per backend, so repeated `.compute()` calls skip recomputation. Chained frames (like after `merge`) reuse the same backend.

## Working with backends

```python
from turbo_frame import TurboFrame, select_backend

tf = TurboFrame(pdf)
tf_cuda = tf.with_backend("cuda")  # force CUDA

print(TurboFrame.available_backends())  # e.g. ["CUDA", "CPU"]

backend = select_backend()  # get the best backend object
tf.compute(backend="cpu")  # override at call time
```

Backends check availability at runtime, so you can share code with people who don't have GPUs—it just falls back to pandas.

## Project layout

```
turbo_frame/
├── __init__.py      # public API
├── _lazy.py         # lazy execution plumbing
├── operations.py    # operation descriptors
├── frame.py         # TurboFrame
├── series.py        # TurboSeries
└── backends/
    ├── __init__.py
    ├── base.py
    ├── cpu.py
    ├── cuda.py
    └── metal.py
```

## Roadmap

- Disk spill for GPUs with limited VRAM
- More string methods to match pandas `.str`
- Benchmark notebooks comparing pandas, cuDF, and MLX

## License

Apache 2.0 — see `LICENSE`.
