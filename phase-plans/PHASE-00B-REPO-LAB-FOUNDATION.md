# PHASE-00B - Repository And Lab Foundation

## Goal

Implement Phase 0 from `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md`: create the clean Python research-engineering foundation for a reproducible Mac-local LLM lab.

## Scope

- Create the initial repo skeleton for configs, data docs, experiments docs, model package, training, evaluation, inference, tokenizer, and tests.
- Add `pyproject.toml` and `uv.lock` for a Python 3.11/3.12 project using PyTorch-first tooling.
- Add deterministic seeding, config loading, dummy fake-token data, checkpoint save/load, and experiment logging conventions.
- Add a tiny dummy training command that runs without private data and writes artifacts only under ignored run directories.
- Add minimal unit tests for config loading, seeding, checkpoint roundtrip, and dummy training.
- Update root docs and `PROGRESS.md`.

## Allowed Paths

- `.gitignore`
- `README.md`
- `NORTH_STAR.md`
- `pyproject.toml`
- `uv.lock`
- `configs/**`
- `data/README.md`
- `docs/**`
- `kgpt/**`
- `tokenizer/**`
- `train/**`
- `eval/**`
- `inference/**`
- `experiments/README.md`
- `tests/**`
- `PROGRESS.md`

## Forbidden Paths

- `.env`
- `.env.*`
- `.venv/**`
- `data/raw/**`
- `data/processed/**`
- `data/tokenized/**`
- `experiments/runs/**`
- `*.pt`
- `*.safetensors`
- `agentic-phase-runner-package/**`
- `automation/**`
- `phase-plans/**`
- `concept-and-ideas/**`
- `runs/**`

## Tasks

- Create the Python package skeleton with minimal, importable modules.
- Add a config format and at least one dummy config under `configs/`.
- Add deterministic seeding helpers for Python, NumPy, and PyTorch.
- Add checkpoint save/load helpers with tests.
- Add a dummy training entrypoint that trains for one epoch on fake token data and writes `config`, `metrics.jsonl`, and checkpoint artifacts under `experiments/runs/`.
- Add docs for local setup, validation commands, data conventions, and experiment artifact conventions.
- Update `PROGRESS.md` with completed work, validation results, and known gaps.

## Acceptance Criteria

- `uv run pytest` passes.
- `uv run ruff check .` passes.
- `git diff --check` passes.
- Dummy training runs from config without private data.
- Checkpoint save/load works in tests.
- Generated data, checkpoints, and run artifacts remain ignored by git.
- README explains how to build the runner, inspect status, and run the first dry-run.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- The foundation can become too abstract if there is no runnable dummy training command.
- PyTorch/MPS dependencies should not be required for CI-like validation beyond importability and dummy CPU tests.

## Out Of Scope

- Character language modeling.
- Real tokenizer training.
- Real Transformer implementation.
- Real dataset collection.
- Long training runs.
