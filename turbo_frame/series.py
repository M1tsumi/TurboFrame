"""TurboSeries: lazy Series facade used by TurboFrame selections."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from ._lazy import LazyObject


def _ensure_series(data: Any, name: str | None = None) -> pd.Series:
    if isinstance(data, pd.Series):
        result = data.copy(deep=False)
    else:
        result = pd.Series(data)
    if name is not None:
        result = result.rename(name)
    return result


class TurboSeries(LazyObject):
    """Lazy Series wrapper primarily produced from TurboFrame column selection."""

    def __init__(self, data: Any, *, name: str | None = None, preferred_backend: str | None = None):
        super().__init__(_ensure_series(data, name=name), preferred_backend=preferred_backend)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def head(self, n: int = 5) -> "TurboSeries":
        return self._queue_operation(
            description=f"head({n})",
            pandas_callable=lambda series, _backend: series.head(n),
        )

    def tail(self, n: int = 5) -> "TurboSeries":
        return self._queue_operation(
            description=f"tail({n})",
            pandas_callable=lambda series, _backend: series.tail(n),
        )

    def sample(self, n: int | None = None, frac: float | None = None, random_state: int | None = None) -> "TurboSeries":
        return self._queue_operation(
            description="sample",
            pandas_callable=lambda series, _backend: series.sample(n=n, frac=frac, random_state=random_state),
        )

    # ------------------------------------------------------------------
    # Element-wise operations
    # ------------------------------------------------------------------
    def map(self, func: Callable[[Any], Any]) -> "TurboSeries":
        return self._queue_operation(
            description="map",
            pandas_callable=lambda series, _backend: series.map(func),
        )

    def apply(self, func: Callable[[Any], Any]) -> "TurboSeries":
        return self._queue_operation(
            description="apply",
            pandas_callable=lambda series, _backend: series.apply(func),
        )

    def fillna(self, value: Any) -> "TurboSeries":
        return self._queue_operation(
            description="fillna",
            pandas_callable=lambda series, _backend: series.fillna(value),
        )

    def replace(self, to_replace: Any, value: Any) -> "TurboSeries":
        return self._queue_operation(
            description="replace",
            pandas_callable=lambda series, _backend: series.replace(to_replace, value),
        )

    def astype(self, dtype: Any) -> "TurboSeries":
        return self._queue_operation(
            description=f"astype({dtype})",
            pandas_callable=lambda series, _backend: series.astype(dtype),
        )

    def clip(self, lower: Any | None = None, upper: Any | None = None) -> "TurboSeries":
        return self._queue_operation(
            description="clip",
            pandas_callable=lambda series, _backend: series.clip(lower=lower, upper=upper),
        )

    def rolling(self, window: int, min_periods: int | None = None) -> "TurboSeries":
        return self._queue_operation(
            description=f"rolling(window={window})",
            pandas_callable=lambda series, _backend: series.rolling(window=window, min_periods=min_periods).mean(),
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def where(self, cond: Callable[[pd.Series], Iterable[bool]] | Sequence[bool]) -> "TurboSeries":
        if callable(cond):
            return self._queue_operation(
                description="where(func)",
                pandas_callable=lambda series, _backend: series[cond(series)],
            )
        mask = list(cond)
        return self._queue_operation(
            description="where(mask)",
            pandas_callable=lambda series, _backend: series[mask],
        )

    def filter(self, cond: Callable[[pd.Series], Iterable[bool]] | Sequence[bool]) -> "TurboSeries":
        return self.where(cond)

    def dropna(self) -> "TurboSeries":
        return self._queue_operation(
            description="dropna",
            pandas_callable=lambda series, _backend: series.dropna(),
        )

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------
    def sort_values(self, ascending: bool = True) -> "TurboSeries":
        return self._queue_operation(
            description="sort_values",
            pandas_callable=lambda series, _backend: series.sort_values(ascending=ascending),
        )

    # ------------------------------------------------------------------
    # String helpers (GPU-friendly via cudf string library)
    # ------------------------------------------------------------------
    def str_lower(self) -> "TurboSeries":
        return self._queue_operation(
            description="str.lower",
            pandas_callable=lambda series, _backend: series.str.lower(),
        )

    def str_upper(self) -> "TurboSeries":
        return self._queue_operation(
            description="str.upper",
            pandas_callable=lambda series, _backend: series.str.upper(),
        )

    def str_contains(self, pattern: str, case: bool = True, regex: bool = True) -> "TurboSeries":
        return self._queue_operation(
            description=f"str.contains('{pattern}')",
            pandas_callable=lambda series, _backend: series.str.contains(pattern, case=case, regex=regex),
        )

    # ------------------------------------------------------------------
    # Materialization helpers
    # ------------------------------------------------------------------
    def to_pandas(self, backend: str | None = None) -> pd.Series:
        result = super().to_pandas(backend)
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result

    @property
    def values(self):
        return self.compute().values

    def sum(self) -> Any:
        return self.compute().sum()

    def mean(self) -> Any:
        return self.compute().mean()

    def min(self) -> Any:
        return self.compute().min()

    def max(self) -> Any:
        return self.compute().max()

    def std(self) -> Any:
        return self.compute().std()

    def var(self) -> Any:
        return self.compute().var()

    def count(self) -> int:
        return int(self.compute().count())

    def nunique(self) -> int:
        return int(self.compute().nunique())

    def to_frame(self, name: str | None = None):
        from .frame import TurboFrame

        series = self.compute()
        return TurboFrame(series.to_frame(name=name or series.name))

    def __getitem__(self, key: Any) -> "TurboSeries":
        return self._queue_operation(
            description="__getitem__",
            pandas_callable=lambda series, _backend: series[key],
        )

    def __repr__(self) -> str:
        plan = " -> ".join(self.describe_plan()) or "<seed>"
        return f"TurboSeries(plan={plan}, pending_ops={len(self.describe_plan())})"


__all__ = ["TurboSeries"]
