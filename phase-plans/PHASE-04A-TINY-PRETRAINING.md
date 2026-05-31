# PHASE-04A - Tiny Pretraining Run

## Goal

Train a small but real LLM from random initialization, roughly 5M to 20M parameters.

## Scope

- Stable training config for a tiny token-level model.
- Learning-rate schedule, gradient clipping, and gradient accumulation.
- Validation loss, perplexity, generated samples, tokens/sec, and memory logging.
- Short experiment report.

## Allowed Paths

- `configs/**`
- `docs/**`
- `kgpt/**`
- `train/**`
- `eval/**`
- `inference/**`
- `tests/**`
- `PROGRESS.md`

## Forbidden Paths

- `.env`
- `.env.*`
- `data/raw/**`
- `data/processed/**`
- `data/tokenized/**`
- `experiments/runs/**`
- `*.pt`
- `*.safetensors`
- `agentic-phase-runner-package/**`
- `automation/**`
- `phase-plans/**`
- `runs/**`

## Tasks

- Add tiny pretraining configs.
- Add metrics and sample logging.
- Add best-checkpoint selection logic.
- Add an experiment report template and one completed tiny-run report.

## Acceptance Criteria

- Validation loss improves meaningfully on a clean small corpus.
- Samples become less random over training.
- The run is reproducible from config.
- The report records data, config, metrics, samples, and failure modes.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- Training artifacts and datasets must remain out of git.
- Loss improvements can reflect leakage or memorization without split checks.

## Out Of Scope

- 30M+ model training.
- Instruction tuning.
