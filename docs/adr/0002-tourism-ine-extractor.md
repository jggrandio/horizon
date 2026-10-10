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
- **past-only covariate** `travellers`. The INE publishes it in the same release as `nights`,
  so its value for a period is unknown when that period is forecast.
- `provisional` carried through as an undeclared column, true when either figure is
  provisional, so evaluation can exclude those periods.
- The period is built from `Anyo` and `T3_Periodo`; omitted months become null rows.

## Consequences

- Domains live under `horizon.domains.<name>`; the core does not import them.
- Using `travellers` at or after a fold's origin leaks the target's release. The layer that
  builds training windows must cut every past-only covariate at the origin; if it does not,
  this covariate is a leak.
- Statistical secrecy nulls both concepts of a series together (checked against the 2026-08
  download: no row has only one of them null, and no series has `nights` entirely null).
  Re-check on refresh: a series whose target is all null passes the contract.
- Forecasting travellers instead means a different target in `load`, not a new extractor.
- `provisional` has no role in the contract yet; the backtesting policy decides how to use it.
