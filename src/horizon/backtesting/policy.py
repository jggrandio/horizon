"""The backtesting policy: generates the folds and cuts each one at its origin."""

from dataclasses import dataclass, replace

import pandas as pd

from horizon.data import TimeSeriesFrame


# eq=False: DataFrame equality is elementwise, so a generated __eq__/__hash__ would break.
@dataclass(frozen=True, slots=True, eq=False)
class Fold:
    """One split. Nothing at or after `origin` reaches `history` or the past-only side of `future`.

    - `history`: every column, periods before `origin`, eligible series only.
    - `future`: model input over the horizon — group key, time index, static and known-future
      covariates. No target, no past-only covariate.
    - `actuals`: group key, time index and target over the horizon, for scoring. Missing
      targets and periods excluded from evaluation are removed.
    - `dropped`: one row per series left out of the fold, group key plus `reason`.
    """

    origin: pd.Timestamp
    history: TimeSeriesFrame
    future: pd.DataFrame
    actuals: pd.DataFrame
    dropped: pd.DataFrame


@dataclass(frozen=True, slots=True)
class BacktestPolicy:
    """Expanding-origin folds: each trains on all history before its origin.

    `origin` is the first evaluation period of the last fold; earlier folds step back from it
    by `step` periods. A series enters a fold only with at least `min_history` observed
    periods before the origin and every known-future covariate present over the horizon.
    Rows where the boolean column `exclude_from_evaluation` is true stay in history and leave
    `actuals` (provisional figures, for instance).
    """

    origin: pd.Timestamp
    horizon: int
    step: int
    folds: int
    min_history: int
    exclude_from_evaluation: str | None = None

    def __post_init__(self) -> None:
        for name in ("horizon", "step", "folds", "min_history"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")

    def split(self, frame: TimeSeriesFrame) -> tuple[Fold, ...]:
        """Folds in ascending order of origin."""
        if pd.date_range(self.origin, periods=1, freq=frame.freq)[0] != self.origin:
            raise ValueError(f"origin {self.origin} is off frequency {frame.freq!r}")
        excluded = self.exclude_from_evaluation
        if excluded is not None:
            if excluded not in frame.data.columns:
                raise ValueError(f"exclusion column {excluded!r} not in data")
            if not pd.api.types.is_bool_dtype(frame.data[excluded]):
                raise ValueError(f"exclusion column {excluded!r} must be boolean")

        span = (self.folds - 1) * self.step + 1
        origins = pd.date_range(end=self.origin, periods=span, freq=frame.freq)[:: self.step]
        return tuple(self._fold(frame, origin) for origin in origins)

    def _fold(self, frame: TimeSeriesFrame, origin: pd.Timestamp) -> Fold:
        data, time, keys = frame.data, frame.time_index, list(frame.group_key)
        known_future = list(frame.known_future_covariates)
        periods = pd.date_range(origin, periods=self.horizon, freq=frame.freq)
        before = data[data[time] < origin]
        window = data[data[time].isin(periods)]

        # Every series over every horizon period, so a gap in the frame shows as a missing covariate.
        series = data[keys].drop_duplicates().reset_index(drop=True)
        future = (
            series.merge(pd.DataFrame({time: periods}), how="cross")
            .merge(window[[*keys, time, *known_future]], on=[*keys, time], how="left")
            .merge(data.drop_duplicates(keys)[[*keys, *frame.static_covariates]], on=keys)
        )

        observed = before.dropna(subset=[frame.target]).groupby(keys).size()
        missing = future[keys].join(future[known_future].isna()).groupby(keys).any()
        status = series.merge(observed.rename("observed").reset_index(), on=keys, how="left")
        status = status.merge(missing.reset_index(), on=keys, how="left")
        flags = pd.DataFrame(
            {
                f"fewer than {self.min_history} observed periods of history": status[
                    "observed"
                ].fillna(0)
                < self.min_history,
                **{
                    f"known-future covariate {c!r} missing in the horizon": status[c]
                    for c in known_future
                },
            }
        )
        status["reason"] = [
            "; ".join(why for why, bad in zip(flags.columns, row, strict=True) if bad)
            for row in flags.itertuples(index=False)
        ]
        kept = pd.MultiIndex.from_frame(status.loc[status["reason"] == "", keys])

        def eligible(df: pd.DataFrame) -> pd.DataFrame:
            in_kept = pd.MultiIndex.from_frame(df[keys]).isin(kept)
            return df[in_kept].sort_values([*keys, time]).reset_index(drop=True)

        scored = window[frame.target].notna()
        if self.exclude_from_evaluation is not None:
            scored &= ~window[self.exclude_from_evaluation]
        actuals = eligible(window.loc[scored, [*keys, time, frame.target]])
        if actuals.empty:
            raise ValueError(
                f"fold at origin {origin.date()} has nothing to evaluate: no eligible series has "
                "an observed, non-excluded target over the horizon"
            )

        return Fold(
            origin=origin,
            history=replace(frame, data=eligible(before)),
            future=eligible(future),
            actuals=actuals,
            dropped=status.loc[status["reason"] != "", [*keys, "reason"]].reset_index(drop=True),
        )
