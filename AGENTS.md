# AGENTS.md

## Project Purpose
This project is a quantitative backtesting system built around vectorbt.
Historical OHLCV data for backtesting is sourced from Binance (chosen for
historical data quality/depth). Bybit is reserved for a later layer:
live/paper-trading execution only, once a strategy has passed backtesting
and validation. Bybit is not used for historical data at any point. The
project is being built incrementally, one capability at a time, across
multiple sessions.

## Tooling Rules (non-negotiable)
- This project uses `uv` exclusively for all environment and package management.
- Use `uv add <package>` for a runtime dependency, `uv add --dev <package>` for
  dev/test dependencies.
- Use `uv run <command>` to execute anything inside the project environment
  (e.g. `uv run pytest`, `uv run python script.py`).
- Never use pip, poetry, or conda, and never hand-edit the dependency sections
  of pyproject.toml.

## General Agent Behavior Rules
- Only build what the current prompt explicitly asks for. Do not add extra
  features, files, folders, or refactor unrelated code "while you're in there."
- If a prompt is ambiguous, or you find yourself about to guess at intent,
  stop and ask instead of assuming.
- Every new piece of logic (function, module, class) must have a corresponding
  test under tests/.
- Before declaring any task complete, run `uv run pytest` and fix any failing
  tests. Never report a task as done with a failing or red test suite.
- Do not introduce new dependencies unless the current prompt explicitly says to.
- Keep changes scoped to one task at a time. Do not silently expand scope
  across multiple layers of the project in a single pass.
- If a task has multiple reasonable approaches (naming, structure, library
  choice) and the prompt doesn't specify, ask before proceeding.

## Project Structure
   quantlab/
   |-- AGENTS.md
   |-- .env.example
   |-- .gitignore
   |-- pyproject.toml
   |-- uv.lock
   |-- src/
   |   `-- quantlab/
   |       |-- __init__.py
   |       |-- data/          (Binance historical OHLCV fetching + caching)
   |       |-- indicators/    (trend and momentum indicators, wrapping vectorbt)
   |       |-- strategies/    (strategy signal generation, e.g. SMA crossover)
   |       |-- backtest/      (vectorbt Portfolio simulation wrapper + metrics)
   |       |-- screening/     (parameter sensitivity sweeps across a strategy's
   |                          parameter grid; in-sample exploratory only, not
   |                          validation)
   |       `-- validation/    (future: walk-forward + Monte Carlo validation)
   |-- tests/
   |   `-- ...                (mirrors src/quantlab/, one test file per module)
   |-- data/                  (local cache of downloaded market data, gitignored)
   `-- scripts/               (manual/runnable entry points, not library
                              code — e.g. demo runs, one-off reports)

   Note: the "(future)" subfolders under src/quantlab/ do not exist yet.
   Only create them when a later prompt explicitly instructs you to build
   that layer.

## Testing Rules
- All tests live under tests/, mirroring the structure of src/quantlab/.
- Test files are named test_<module>.py.
- Run the full suite with `uv run pytest` before ending any task.
- A task is not complete until the full suite passes with zero failures.

## Completion Reporting Format
At the end of every task, report back using exactly this structure:

- **Files created:** bullet list of new files, each with a one-line purpose
- **Files modified:** bullet list of edited files, each with a one-line
  description of what changed
- **Tests added:** bullet list of new test cases, each with a one-line
  description of what it verifies
- **Test run output:** the full, verbatim output of `uv run pytest`
- **Manual run output (if applicable):** if the task produces a
  runnable script or CLI entry point, actually execute it for real
  (real data, real network calls where relevant — not mocked) and
  include the full, verbatim console output. This is separate from and
  in addition to the automated pytest output.
- **Scope check:** explicit confirmation that no files or folders were
  created outside what the current prompt specified, and that no unrelated
  code was modified
- **Claim verification:** any narrative statement describing what changed,
  what was affected, or what the results show (e.g. in "Comparison &
  Analysis" or similar sections) must be checked line-by-line against the
  actual before/after data before being included in the report. Do not
  describe expected/intended behavior as if it were confirmed — confirm it
  against the real printed numbers first. If actual results don't match
  what the code change intended, report the discrepancy explicitly rather
  than describing the intended behavior.
- **Assumptions made:** bullet list of any judgment calls made where the
  prompt didn't fully specify something (e.g. a chosen exception type, a
  naming choice) — even small ones
- **Open items:** anything noticed but not acted on because it was out of
  scope for this task (edge cases, potential issues, follow-up ideas)

If a task could not be fully completed, use this same structure but state
clearly what is blocking completion instead of a Test run output.

## Strategy Testing Mode
This entire section only applies when the user's message begins with the
exact literal text "@strategy" as the very first characters of the prompt.
If "@strategy" does not appear at the very start of the message (e.g. it
appears mid-sentence or at the end), ignore this section completely and
treat the prompt as a normal Build Mode task under the rules above.

(This section is a placeholder. Its detailed rules for walk-forward
validation, Monte Carlo simulation, and strategy acceptance criteria will
be added in a later prompt, once that layer of the project is built. Do
not invent strategy-testing behavior on your own in the meantime.)

## Backlog / Future Layers
This section tracks capabilities intentionally deferred to future sessions,
so they survive across chat windows with no shared memory. When a layer is
deferred, add an item here. When it's later built, remove its entry.

Current backlog:
- Expand extract_metrics() in src/quantlab/backtest/engine.py with
  additional metrics (e.g. Sortino ratio, Calmar ratio, profit factor,
  drawdown duration) once a future validation layer needs them.
- Expand indicator suite beyond SMA/EMA/RSI (e.g. MACD, Bollinger Bands,
  ATR, Stochastic) using the same vectorbt-wrapper pattern established in
  src/quantlab/indicators, once needed by a specific strategy.
- Incremental/partial cache updates for src/quantlab/data/ohlcv.py
  (currently only supports one full-range cache file per
  symbol/timeframe/range; fetching only the missing days of an
  already-partially-cached range is not yet supported).
- Walk-forward validation harness (rolling training windows, out-of-sample
  testing, aggregation across folds).
- Monte Carlo simulation on trade sequences/returns.
- Config/settings module for shared constants (default symbol, timeframe,
  fee/slippage assumptions) if hardcoded values start duplicating across
  modules.
- Bybit connector for live/paper-trading execution (only after a strategy
  has passed backtesting and validation).

