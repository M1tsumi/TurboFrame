"""TurboFrame: a lazy, backend-aware DataFrame facade."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from ._lazy import LazyObject

if pd.__version__ < "1.5":  # pragma: no cover - sanity guard
    raise RuntimeError("TurboFrame requires pandas >= 1.5")

DataFrameLike = pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]


def _ensure_dataframe(data: DataFrameLike | pd.Series | "TurboFrame") -> pd.DataFrame:
    if isinstance(data, TurboFrame):  # type: ignore[name-defined]
        return data.compute()
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=False)
    return pd.DataFrame(data)


class TurboFrame(LazyObject):
    """Lazy DataFrame wrapper that records transformations before execution."""

    def __init__(self, data: DataFrameLike | pd.DataFrame, *, preferred_backend: str | None = None):
        super().__init__(_ensure_dataframe(data), preferred_backend=preferred_backend)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, *, preferred_backend: str | None = None) -> "TurboFrame":
        return cls(df, preferred_backend=preferred_backend)

    @classmethod
    def read_csv(cls, path: str, **kwargs) -> "TurboFrame":
        return cls(pd.read_csv(path, **kwargs))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def head(self, n: int = 5) -> "TurboFrame":
        return self._queue_operation(
            description=f"head({n})",
            pandas_callable=lambda df, _backend: df.head(n),
        )

    def tail(self, n: int = 5) -> "TurboFrame":
        return self._queue_operation(
            description=f"tail({n})",
            pandas_callable=lambda df, _backend: df.tail(n),
        )

    def sample(self, n: int | None = None, frac: float | None = None, random_state: int | None = None) -> "TurboFrame":
        return self._queue_operation(
            description="sample",
            pandas_callable=lambda df, _backend: df.sample(n=n, frac=frac, random_state=random_state),
        )

    # ------------------------------------------------------------------
    # Column selection & projection
    # ------------------------------------------------------------------
    def select(self, columns: Sequence[str]) -> "TurboFrame":
        cols = list(columns)
        return self._queue_operation(
            description=f"select({cols})",
            pandas_callable=lambda df, _backend: df.loc[:, cols],
        )

    def drop(self, columns: Sequence[str]) -> "TurboFrame":
        cols = list(columns)
        return self._queue_operation(
            description=f"drop({cols})",
            pandas_callable=lambda df, _backend: df.drop(columns=cols),
        )

    def rename(self, columns: Mapping[str, str] | None = None) -> "TurboFrame":
        rename_map = dict(columns or {})
        return self._queue_operation(
            description=f"rename({rename_map})",
            pandas_callable=lambda df, _backend: df.rename(columns=rename_map),
        )

    def assign(self, **new_columns: Callable[[pd.DataFrame], Any] | Any) -> "TurboFrame":
        return self._queue_operation(
            description=f"assign({list(new_columns)})",
            pandas_callable=lambda df, _backend: df.assign(**new_columns),
        )

    def with_columns(self, mapping: Mapping[str, Any]) -> "TurboFrame":
        payload = dict(mapping)
        return self._queue_operation(
            description=f"with_columns({list(payload)})",
            pandas_callable=lambda df, _backend: df.assign(**payload),
        )

    def astype(self, dtype: Mapping[str, Any] | str) -> "TurboFrame":
        return self._queue_operation(
            description=f"astype({dtype})",
            pandas_callable=lambda df, _backend: df.astype(dtype),
        )

    # ------------------------------------------------------------------
    # Filtering & transformations
    # ------------------------------------------------------------------
    def filter(self, predicate: str | Callable[[pd.DataFrame], Any]) -> "TurboFrame":
        if isinstance(predicate, str):
            return self.query(predicate)

        def _call(df: pd.DataFrame, _backend):
            mask = predicate(df)
            return df.loc[mask]

        return self._queue_operation(description="filter", pandas_callable=_call)

    def query(self, expr: str, **kwargs) -> "TurboFrame":
        return self._queue_operation(
            description=f"query('{expr}')",
            pandas_callable=lambda df, _backend: df.query(expr, **kwargs),
        )

    def dropna(self, how: str = "any", subset: Sequence[str] | None = None) -> "TurboFrame":
        return self._queue_operation(
            description=f"dropna({how})",
            pandas_callable=lambda df, _backend: df.dropna(how=how, subset=subset),
        )

    def fillna(self, value: Any) -> "TurboFrame":
        return self._queue_operation(
            description="fillna",
            pandas_callable=lambda df, _backend: df.fillna(value),
        )

    def replace(self, to_replace: Any, value: Any) -> "TurboFrame":
        return self._queue_operation(
            description="replace",
            pandas_callable=lambda df, _backend: df.replace(to_replace, value),
        )

    def apply(self, func: Callable[[pd.DataFrame], pd.Series] | Callable[[pd.Series], Any], axis: int = 0) -> "TurboFrame":
        return self._queue_operation(
            description="apply",
            pandas_callable=lambda df, _backend: df.apply(func, axis=axis),
        )

    def map_column(self, column: str, func: Callable[[pd.Series], Any], result_name: str | None = None) -> "TurboFrame":
        new_name = result_name or column

        def _call(df: pd.DataFrame, _backend):
            df = df.copy(deep=False)
            df[new_name] = df[column].map(func)
            return df

        return self._queue_operation(description=f"map_column({column}->{new_name})", pandas_callable=_call)

    # ------------------------------------------------------------------
    # Sorting & windowing
    # ------------------------------------------------------------------
    def sort_values(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        na_position: str = "last",
    ) -> "TurboFrame":
        return self._queue_operation(
            description=f"sort_values({by})",
            pandas_callable=lambda df, _backend: df.sort_values(by=by, ascending=ascending, na_position=na_position),
        )

    def sort_index(self, ascending: bool = True) -> "TurboFrame":
        return self._queue_operation(
            description="sort_index",
            pandas_callable=lambda df, _backend: df.sort_index(ascending=ascending),
        )

    # ------------------------------------------------------------------
    # Aggregations & groupby
    # ------------------------------------------------------------------
    def aggregate(self, agg: Mapping[str, Any] | Sequence[Any] | Callable[[pd.DataFrame], Any]) -> "TurboFrame":
        return self._queue_operation(
            description=f"aggregate({agg})",
            pandas_callable=lambda df, _backend: df.aggregate(agg),
        )

    def groupby_agg(
        self,
        by: str | Sequence[str],
        agg: Mapping[str, Any] | Sequence[Any] | Callable[[pd.DataFrame], Any],
        dropna: bool = True,
        sort: bool = False,
    ) -> "TurboFrame":
        def _call(df: pd.DataFrame, _backend):
            grouped = df.groupby(by=by, dropna=dropna, sort=sort)
            result = grouped.agg(agg)
            return result.reset_index()

        return self._queue_operation(description=f"groupby({by})", pandas_callable=_call)

    # ------------------------------------------------------------------
    # Joins & concatenation
    # ------------------------------------------------------------------
    def merge(
        self,
        other: "TurboFrame" | pd.DataFrame | Mapping[str, Sequence[Any]],
        *,
        how: str = "inner",
        on: str | Sequence[str] | None = None,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> "TurboFrame":
        def _call(df: pd.DataFrame, backend):
            other_df = _ensure_dataframe(other)
            if isinstance(other, TurboFrame):
                other_df = other._materialize_with_backend(backend)
            return df.merge(
                other_df,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                suffixes=suffixes,
            )

        return self._queue_operation(description=f"merge({how})", pandas_callable=_call)

    def concat(self, others: Iterable["TurboFrame" | pd.DataFrame], axis: int = 0) -> "TurboFrame":
        others_list = list(others)

        def _call(df: pd.DataFrame, backend):
            materialized = [df]
            for other in others_list:
                if isinstance(other, TurboFrame):
                    materialized.append(other._materialize_with_backend(backend))
                else:
                    materialized.append(_ensure_dataframe(other))
            return pd.concat(materialized, axis=axis)

        return self._queue_operation(description=f"concat(axis={axis})", pandas_callable=_call)

    # ------------------------------------------------------------------
    # Column accessors
    # ------------------------------------------------------------------
    def __getitem__(self, key: str | Sequence[str]):
        if isinstance(key, str):
            from .series import TurboSeries

            return self._queue_operation(
                description=f"column('{key}')",
                pandas_callable=lambda df, _backend: df[key],
                result_cls=TurboSeries,
            )
        return self.select(key)

    def get_series(self, column: str) -> "TurboSeries":
        return self[column]

    # ------------------------------------------------------------------
    # Materialization helpers
    # ------------------------------------------------------------------
    def to_pandas(self, backend: str | None = None) -> pd.DataFrame:
        return super().to_pandas(backend)

    def __repr__(self) -> str:
        plan = " -> ".join(self.describe_plan()) or "<seed>"
        return f"TurboFrame(plan={plan}, pending_ops={len(self.describe_plan())})"


__all__ = ["TurboFrame"]
