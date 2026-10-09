# 0001. Data contract: TimeSeriesFrame

## Context

Training, inference and backtesting must work for any domain without being edited. They
need one shape of input that says which column plays which role.

## Decision

`horizon.data.TimeSeriesFrame`: a frozen dataclass holding a long-format DataFrame (one row
per group key and period) plus the target, time index, frequency, group key and the three
kinds of covariate. It wraps the DataFrame rather than subclassing it, because pandas
subclasses lose their metadata through most operations.

### Validation

- **Extractor** (per domain): cleans the source — builds the period, turns omitted rows into null rows — because only it knows the source's quirks.
- **Contract** (`TimeSeriesFrame.__post_init__`): checks structure — declared columns exist with one role each, dtypes, unique periods, every period on `freq` (gaps allowed until the missing-data study), static covariates constant — because these hold for every domain and the core relies on them.
- **Task** (where the horizon is set): checks that known-future covariates reach the last observation plus the horizon, because the horizon is a property of the task, not of the data.

## Consequences

- Adding a domain means writing an extractor that produces a valid `TimeSeriesFrame`.
- Row order is not guaranteed; code that needs time order sorts first.
- The DataFrame inside is still mutable; code that changes `.data` bypasses validation.
- Gaps are allowed, so downstream code cannot assume one row per period; a later ADR
  settles this after the missing-data study.
- The frequency check loops per series, which is fine at INE scale but slow for very many series.
