# 0004. Backtesting policy: expanding origin, cut at the origin

## Context

The horizon, the minimum length per series, the known-future check (ADR 0001) and the cut of
past-only covariates (ADR 0002) all need an origin and a horizon, which the data contract
does not have. Folds must be generated, never supplied. The INE panel has 21 years with the
pandemic in the middle, series that start late, and provisional figures.

## Decision

`horizon.backtesting.BacktestPolicy(origin, horizon, step, folds, min_history,
exclude_from_evaluation=None)`. `split(frame)` returns one `Fold` per origin, ascending.
`origin` is the first evaluation period of the last fold; earlier folds step back by `step`.

- **Expanding origin.** Each fold trains on everything before its origin. A sliding window
  is a truncation the model can apply to an expanding history, but the reverse is not
  possible, so how much history to use is the model's choice. Folds then differ only in
  their origin, so errors are comparable across them; with a fixed window the pandemic's
  share of training would change from fold to fold.
- **The fold carries cut frames, not indices.** Each fold holds `history` (a
  `TimeSeriesFrame`, every column, before the origin), `future` (group key, time index,
  static and known-future covariates over the horizon; no target, no past-only covariate)
  and `actuals` (target over the horizon). With indices, the past-only cut would be left
  to the caller and the leak test could not see it. That means one copy per fold, not per
  series. A global model uses the frames as they are; a per-series model groups them.
- **Series that fall short are dropped and reported.** A series enters a fold with at
  least `min_history` observed targets before the origin and every known-future covariate
  present over the horizon. Otherwise it leaves all three frames and appears in
  `fold.dropped` with the reason. Rejecting would make early folds unusable; dropping
  silently hides which series a score covers.
- **Provisional figures stay in history and leave evaluation.** The policy takes the name
  of a boolean column. A fold with nothing left to evaluate raises.
  The contract gains no role for it: the contract describes data, not evaluation.

## Consequences

- The leak tests use a panel whose values equal their time position, so a leak shows up as
  arithmetic, whatever the implementation does. A second test nulls the target while the
  past-only covariate keeps arriving, which catches a cut made on the target alone.
- History holds revised figures where the forecaster at the time saw first releases. Only
  the latest vintage is downloaded, so backtests are slightly optimistic.
- A series with enough history but no target over the horizon (a discontinued point) stays
  in `future` and simply contributes nothing to scoring.
- With the 2026-08 download: 276 series. 64 of them start in 2024 and are dropped from
  every fold that needs 36 months of history.
- `split` builds the frames for every fold eagerly: about 3 s for six folds on table 2078.
