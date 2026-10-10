# 0003. Enforce the dependency rule in CI

## Context

The core (`horizon.data`, `training`, `inference`, `backtesting`) must not import domains or
modelling frameworks. Until now only `make verify` run by hand stood between a change and
`main`, and nothing in it checked imports.

## Decision

- `import-linter` holds two forbidden contracts in `pyproject.toml`: the core does not
  import `horizon.domains`, and the core does not import `torch`, `lightgbm` or
  `pytorch_forecasting`. Indirect imports count.
- `make verify` runs `lint-imports` alongside ruff, mypy and pytest.
- A GitHub Actions workflow runs `make verify` on every push to `main` and every PR. Its
  `verify` job is a required check on `main`.

## Consequences

- A PR that crosses the boundary cannot be merged, even by the maintainer.
- `training`, `inference` and `backtesting` exist as empty packages so the contracts cover
  them from the start; import-linter rejects contracts naming modules that do not exist.
- A new core package must be added to both contracts' `source_modules`. A new framework
  goes in the framework contract's `forbidden_modules`.
- CI uses `uv sync --locked`, so a `pyproject.toml` change without a refreshed `uv.lock`
  fails.
