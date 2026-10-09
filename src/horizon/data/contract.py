"""The data contract: a long-format DataFrame plus the roles of its columns."""

from dataclasses import dataclass

import pandas as pd


# eq=False: DataFrame equality is elementwise, so a generated __eq__/__hash__ would break.
@dataclass(frozen=True, slots=True, eq=False)
class TimeSeriesFrame:
    """One row per (group key, period). Validated on construction.

    The target may be null (missing observation, or a future period that carries only
    known-future covariates). Every period must fall on `freq`; series may have gaps.
    """

    data: pd.DataFrame
    target: str
    time_index: str
    freq: str
    group_key: tuple[str, ...]
    static_covariates: tuple[str, ...] = ()
    known_future_covariates: tuple[str, ...] = ()
    past_only_covariates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.group_key:
            raise ValueError("group key is empty; a single series still needs a key column")

        declared = [
            self.target,
            self.time_index,
            *self.group_key,
            *self.static_covariates,
            *self.known_future_covariates,
            *self.past_only_covariates,
        ]
        repeated = sorted({c for c in declared if declared.count(c) > 1})
        if repeated:
            raise ValueError(f"columns declared more than once: {repeated}")
        missing = [c for c in declared if c not in self.data.columns]
        if missing:
            raise ValueError(f"declared columns not in data: {missing}")

        data = self.data
        keys = [*self.group_key, self.time_index]
        if data[keys].isna().any().any():
            raise ValueError(f"group key or time index contains null: {keys}")
        if not pd.api.types.is_datetime64_any_dtype(data[self.time_index]):
            raise ValueError(f"time index {self.time_index!r} must be datetime64")
        if not pd.api.types.is_numeric_dtype(data[self.target]):
            raise ValueError(f"target {self.target!r} must be numeric")

        if data.duplicated(keys).any():
            raise ValueError(f"duplicate periods within a series on {keys}")

        for key, times in data.groupby(list(self.group_key), sort=False)[self.time_index]:
            expected = pd.date_range(times.min(), times.max(), freq=self.freq)
            # shortcut: gaps pass until the missing-data study decides between null rows and rejecting.
            if not pd.DatetimeIndex(times).isin(expected).all():
                raise ValueError(f"series {key} has a period off frequency {self.freq!r}")

        if self.static_covariates:
            varying = data.groupby(list(self.group_key))[list(self.static_covariates)].nunique(
                dropna=False
            )
            bad = varying.columns[varying.gt(1).any()].tolist()
            if bad:
                raise ValueError(f"static covariate varies within a series: {bad}")
