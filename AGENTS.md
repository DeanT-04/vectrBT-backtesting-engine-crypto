# AGENTS.md

## Project Purpose
This project is a quantitative backtesting system built around vectorbt,
eventually connecting to Bybit (falling back to Binance for historical data
where Bybit lacks it) for market data. It is being built incrementally, one
capability at a time, across multiple sessions.

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
   |       |-- data/          (future: exchange connectors, data loading/caching)
   |       |-- indicators/    (future: signal/indicator logic)
   |       |-- strategies/    (future: strategy definitions)
   |       |-- backtest/      (future: vectorbt simulation wrappers)
   |       `-- validation/    (future: walk-forward + Monte Carlo validation)
   |-- tests/
   |   `-- ...                (mirrors src/quantlab/, one test file per module)
   `-- data/                  (local cache of downloaded market data, gitignored)

   Note: the "(future)" subfolders under src/quantlab/ do not exist yet.
   Only create them when a later prompt explicitly instructs you to build
   that layer.

## Testing Rules
- All tests live under tests/, mirroring the structure of src/quantlab/.
- Test files are named test_<module>.py.
- Run the full suite with `uv run pytest` before ending any task.
- A task is not complete until the full suite passes with zero failures.

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
