# PHASE-04A - Tiny Pretraining Run

## Goal

Train a small but real LLM from random initialization, roughly 5M to 20M parameters.

## Scope

- Stable training config for a tiny token-level model.
- Learning-rate schedule, gradient clipping, and gradient accumulation.
- Validation loss, perplexity, generated samples, tokens/sec, and memory logging.
- Short experiment report.

## Prerequisites

- PHASE-03A is complete.
- Tokenizer/data manifest from PHASE-02A exists and has approved source/license notes.
- Fixed prompt/eval assets from earlier phases are available for sample tracking.

## Phase Dependencies

- Depends on PHASE-03A Transformer and PHASE-02A tokenizer/data pipeline.
- Unblocks PHASE-05A by producing tiny pretraining evidence, a scaling baseline, and fixed-prompt samples.

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

These paths are forbidden to commit or manually edit as phase source changes. Required commands may generate ignored evidence artifacts only when this phase's artifact policy allows them.

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

- Select a small curated dataset with documented source/license, split, dedup, and leakage checks.
- Add tiny pretraining configs for a 5M-20M parameter target, including model config, optimizer, scheduler, batch size, gradient accumulation, context length, checkpoint cadence, validation cadence, sample cadence, and run budget.
- Define run budget in steps/tokens/wall-clock target and hardware/device.
- Add metrics and sample logging.
- Add best-checkpoint selection logic.
- Add an experiment report template and one completed tiny-run report.
- Add resume behavior for interrupted training and test or smoke-test checkpoint resume.
- Define meaningful loss improvement before running: default is validation loss improves by at least 10% over the first stable validation point, unless a smaller threshold is justified in the report.

## Deliverables

- Tiny pretraining config.
- Training/resume path for token-level pretraining.
- Best-checkpoint selection.
- Completed experiment report.
- Updated fixed prompt sample set if needed.

## Evidence Artifacts

- Ignored run directory with config, metrics, samples, checkpoint_last, checkpoint_best, tokenizer_info, eval report, and manifest.
- Report sections: dataset source/license, tokenizer, model config, parameter count, run budget, hardware/device, hyperparameters, validation curve summary, sample progression, failure modes, resume behavior, and next scale recommendation.
- Checkpoint metadata for last and best checkpoints.

## Artifact Policy

- Dataset files, tokenized files, checkpoints, and generated run outputs remain ignored.
- Commit only configs, code, tests, and report summaries with paths to ignored artifacts.

## Acceptance Criteria

- Validation loss improves by the predeclared threshold on a clean small corpus, or the phase is marked blocked with diagnostic evidence.
- Samples become less random over training.
- The run is reproducible from config.
- The report records data, config, metrics, samples, and failure modes.
- Resume behavior is implemented and documented.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke`
- `uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --checkpoint experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt --output docs/phase04a_tiny_report.md`

## Human Decisions

- Approve dataset choice/license before using results as project evidence.
- Approve whether the loss improvement and samples justify moving to 30M+ training.

## Phase Gate

Mark complete only when a 5M-20M run produces a best checkpoint, reproducible run artifacts, fixed-prompt samples, and a report that supports or blocks scale-up.

## Risks

- Training artifacts and datasets must remain out of git.
- Loss improvements can reflect leakage or memorization without split checks.

## Out Of Scope

- 30M+ model training.
- Instruction tuning.

## Deferred Backlog

- Larger data mixtures and kgpt-30m/50m/100m configs move to PHASE-05A.
