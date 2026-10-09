# horizon
 
Time-series forecasting with a reusable core and swappable domains. First domain: Spanish
tourism (INE hotel occupancy). Public repo, single maintainer.
 
## The dependency rule
 
The core (`src/horizon/data`, `training`, `backtesting`) imports no modelling framework.
No torch, no lightgbm, no pytorch-forecasting. Those live behind adapters.
 
This is the project's central claim, so it is enforced, not trusted: `import-linter`
contracts run in CI. When a change needs a framework import in the core, the answer is an
adapter, not an exception.
 
Adding a domain means writing one extractor that maps a source onto the data contract.
Training, inference and backtesting stay untouched. If a change would require editing them
to support a new domain, the contract is wrong — say so rather than widening the core.
 
## Vocabulary
 
Use these words in code, errors, docs and commits. They mean the same thing everywhere.
 
- **series** — one time-ordered sequence of the target, identified by its group key.
- **group key** — the columns that identify a series.
- **time index** — the period column, at a declared frequency.
- **horizon** — how many periods ahead a forecast covers. A property of the task.
- **static covariate** — constant within a group.
- **known-future covariate** — available for every period up to the last observation plus
  the horizon. The validator checks this; a covariate that falls short is a leak.
- **past-only covariate** — observed, not available at prediction time.
- **backtesting policy** — origin, horizon, step, number of folds. Generates the splits;
  splits are never supplied from outside.
## Data
 
Downloads go to `data/`, which is gitignored. The repo holds extraction scripts; the data
itself stays out of version control and out of CI.
 
Source terms, which the README repeats for readers:
 
- **INE** — reuse permitted including commercially, with attribution ("Elaboración propia
  con datos extraídos del sitio web del INE: www.ine.es"). No API key.
- **Antonio, Almeida & Nunes (2019)** — CC BY, cite the paper.
### INE gotchas, learned the hard way
 
- Table **2078** (tourist points, 2005-01 onward) is the source. Table 75197 is the same
  content starting 2025-01 — too short for seasonality.
- Build the period from `Anyo` and `T3_Periodo`. The `Fecha` field carries a timezone
  offset that shifts months across DST.
- `Data` arrives newest-first. Sort it.
- Missing values appear as a row with `Valor: null` in some tables and as an omitted row in
  others. Extractors normalise both to the same thing.
- `T3_TipoDato` marks provisional figures, which the INE later revises. Carry the column
  through and exclude provisional periods from evaluation.
- The dimension name is misspelled upstream as `PUNTOS TURISTÍCOS`. Match it literally.
 
## Conventions
 
- Tests before implementation in the core, where there is a contract to test against.
  Exploration and model tuning do not get this treatment.
- The leak test is the one that matters most: no fold may see data after its origin.
- Frozen dataclasses with `slots=True` for specifications; `tuple[...]`, not `list[...]`,
  so a frozen object is genuinely frozen.
- Write the second implementation before extracting an interface.
- Notebooks call into `src` and plot. A function that appears in a notebook moves to `src`
  and gets imported back.
- One PR per decision, squash-merged into `main`, with the ADR in the same PR.
- Commits and PR titles follow Conventional Commits: `type(scope): summary`, with type one
  of `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`. Scope is the package
  or domain (`data`, `training`, `tourism`), optional.
- `make verify` (lint, format check, mypy, tests) passes before every commit.
## Decisions
 
`docs/adr/` holds one short file per decision: context, decision, consequences. They are
immutable — supersede with a new ADR rather than editing an old one. Read them before
proposing a change to anything they cover.
 
`docs/design.md` is the design doc. It goes stale; the ADRs do not.
