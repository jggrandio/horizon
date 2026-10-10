# 0002. Tourism domain: INE table 2078 extractor

## Context

The first domain needs an extractor that maps INE table 2078 (hotel travellers and nights
per tourist point, monthly, 2005-01 onward) onto `TimeSeriesFrame`. The table arrives as
one JSON series per (point, concept, residence).

## Decision

`horizon.domains.tourism.ine` downloads the raw JSON once (cached under `data/`) and
pivots it to one row per (point, residence, period):

- **group key** `(point, residence)`, using the INE names.
- **target** `nights` (pernoctaciones), the usual occupancy measure.
- **past-only covariate** `travellers`, observed alongside the target.
- `provisional` carried through as an undeclared column, true when either figure is
  provisional, so evaluation can exclude those periods.
- The period is built from `Anyo` and `T3_Periodo`; omitted months become null rows.

## Consequences

- Domains live under `horizon.domains.<name>`; the core does not import them.
- Forecasting travellers instead means a different target in `load`, not a new extractor.
- `provisional` has no role in the contract yet; the backtesting policy decides how to use it.
