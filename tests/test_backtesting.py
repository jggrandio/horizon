import numpy as np
import pandas as pd
import pytest

from horizon.backtesting import BacktestPolicy
from horizon.data import TimeSeriesFrame

START = pd.Timestamp("2020-01-01")
PERIODS = 36


def make_panel(starts: dict[str, int] | None = None) -> TimeSeriesFrame:
    """Every value is its own time position, so any leak is arithmetically visible."""
    starts = starts or {"a": 0, "b": 0, "c": 0}
    rows = []
    for key, first in starts.items():
        for t in range(first, PERIODS):
            rows.append(
                {
                    "point": key,
                    "period": START + pd.DateOffset(months=t),
                    "nights": float(t),
                    "travellers": float(t),
                    "holidays": float(t),
                    "region": key.upper(),
                    "provisional": False,
                }
            )
    return TimeSeriesFrame(
        pd.DataFrame(rows),
        target="nights",
        time_index="period",
        freq="MS",
        group_key=("point",),
        static_covariates=("region",),
        known_future_covariates=("holidays",),
        past_only_covariates=("travellers",),
    )


def month(t: int) -> pd.Timestamp:
    return START + pd.DateOffset(months=t)


def policy(**overrides: object) -> BacktestPolicy:
    kwargs: dict[str, object] = {
        "origin": month(30),
        "horizon": 3,
        "step": 2,
        "folds": 4,
        "min_history": 12,
    }
    return BacktestPolicy(**{**kwargs, **overrides})  # type: ignore[arg-type]


def test_no_fold_sees_data_after_its_origin() -> None:
    for fold in policy().split(make_panel()):
        history = fold.history.data
        assert history["nights"].max() < fold.actuals["nights"].min()


def test_past_only_covariates_are_cut_at_the_origin() -> None:
    frame = make_panel()
    # The trap: target missing from t=20 on while the covariate keeps arriving.
    frame.data.loc[frame.data["nights"] >= 20, "nights"] = np.nan
    for fold in policy(origin=month(18), folds=2).split(frame):
        assert fold.history.data["travellers"].max() < fold.actuals["nights"].min()
        assert "travellers" not in fold.future.columns
        assert "nights" not in fold.future.columns


def test_future_carries_known_future_and_static_covariates_for_the_horizon() -> None:
    fold = policy(folds=1).split(make_panel())[0]
    assert set(fold.future.columns) == {"point", "period", "region", "holidays"}
    assert sorted(fold.future["holidays"].unique()) == [30.0, 31.0, 32.0]
    assert len(fold.future) == 3 * 3


def test_origins_step_back_from_the_latest() -> None:
    folds = policy().split(make_panel())
    assert [f.origin for f in folds] == [month(24), month(26), month(28), month(30)]


def test_history_is_expanding() -> None:
    for fold in policy().split(make_panel()):
        assert fold.history.data["period"].min() == START


def test_short_series_is_dropped_and_reported() -> None:
    # "c" starts at t=14: 12 periods of history only from origin t=26 on.
    folds = policy().split(make_panel({"a": 0, "b": 0, "c": 14}))
    reported = {f.origin: set(f.dropped["point"]) for f in folds}
    assert reported == {month(24): {"c"}, month(26): set(), month(28): set(), month(30): set()}
    first = folds[0]
    assert "c" not in set(first.history.data["point"]) | set(first.future["point"])
    assert first.dropped["reason"].str.contains("history").all()


def test_series_short_of_known_future_covariate_is_dropped_and_reported() -> None:
    frame = make_panel()
    frame.data.loc[
        (frame.data["point"] == "b") & (frame.data["period"] == month(31)), "holidays"
    ] = np.nan
    fold = policy(folds=1).split(frame)[0]
    assert list(fold.dropped["point"]) == ["b"]
    assert fold.dropped["reason"].str.contains("holidays").all()
    assert "b" not in set(fold.future["point"])


def test_excluded_periods_stay_in_history_and_leave_evaluation() -> None:
    frame = make_panel()
    frame.data["provisional"] = frame.data["period"] >= month(32)
    last = policy(exclude_from_evaluation="provisional").split(frame)[-1]
    assert list(last.actuals["nights"]) == [30.0, 31.0] * 3
    later = policy(origin=month(33), folds=1, exclude_from_evaluation="provisional")
    with pytest.raises(ValueError, match="nothing to evaluate"):
        later.split(frame)


def test_excluded_periods_in_history_are_kept() -> None:
    frame = make_panel()
    frame.data["provisional"] = frame.data["period"] == month(10)
    fold = policy(folds=1, exclude_from_evaluation="provisional").split(frame)[0]
    assert fold.history.data["provisional"].sum() == 3


def test_origin_off_frequency_is_rejected() -> None:
    with pytest.raises(ValueError, match="frequency"):
        policy(origin=pd.Timestamp("2022-07-15")).split(make_panel())


def test_unknown_exclusion_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in data"):
        policy(exclude_from_evaluation="revised").split(make_panel())


@pytest.mark.parametrize("field", ["horizon", "step", "folds", "min_history"])
def test_non_positive_settings_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        policy(**{field: 0})
