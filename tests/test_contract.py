import dataclasses

import pandas as pd
import pytest

from horizon.data import TimeSeriesFrame


def make_data() -> pd.DataFrame:
    periods = pd.date_range("2024-01-01", periods=3, freq="MS")
    return pd.DataFrame(
        {
            "point": ["a"] * 3 + ["b"] * 3,
            "period": list(periods) * 2,
            "nights": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "region": ["north"] * 3 + ["south"] * 3,
            "holidays": [0, 1, 0, 2, 0, 1],
            "rooms": [10, 11, 12, 20, 21, 22],
        }
    )


def make(data: pd.DataFrame | None = None, **overrides: object) -> TimeSeriesFrame:
    kwargs: dict[str, object] = {
        "data": make_data() if data is None else data,
        "target": "nights",
        "time_index": "period",
        "freq": "MS",
        "group_key": ("point",),
        "static_covariates": ("region",),
        "known_future_covariates": ("holidays",),
        "past_only_covariates": ("rooms",),
    }
    return TimeSeriesFrame(**{**kwargs, **overrides})  # type: ignore[arg-type]


def test_valid_frame_is_frozen() -> None:
    frame = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.target = "rooms"  # type: ignore[misc]


def test_missing_target_is_allowed() -> None:
    assert make().data["nights"].isna().sum() == 1


def test_undeclared_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in data"):
        make(past_only_covariates=("occupancy",))


def test_column_declared_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        make(past_only_covariates=("holidays",))


def test_empty_group_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="group key"):
        make(group_key=())


def test_time_index_must_be_datetime() -> None:
    data = make_data()
    data["period"] = data["period"].dt.strftime("%Y-%m")
    with pytest.raises(ValueError, match="datetime"):
        make(data)


def test_target_must_be_numeric() -> None:
    with pytest.raises(ValueError, match="numeric"):
        make(target="region", static_covariates=())


def test_duplicate_period_in_series_is_rejected() -> None:
    data = make_data()
    data.loc[1, "period"] = data.loc[0, "period"]
    with pytest.raises(ValueError, match="duplicate"):
        make(data)


def test_unsorted_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        make(make_data().iloc[::-1].reset_index(drop=True))


def test_gap_in_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="gap"):
        make(make_data().drop(index=1).reset_index(drop=True))


def test_period_off_frequency_is_rejected() -> None:
    data = make_data()
    data["period"] = data["period"] + pd.Timedelta(days=14)
    with pytest.raises(ValueError, match="off frequency"):
        make(data)


def test_varying_static_covariate_is_rejected() -> None:
    data = make_data()
    data.loc[1, "region"] = "east"
    with pytest.raises(ValueError, match="static covariate"):
        make(data)


def test_null_in_group_key_is_rejected() -> None:
    data = make_data()
    data.loc[0, "point"] = None
    with pytest.raises(ValueError, match="null"):
        make(data)
