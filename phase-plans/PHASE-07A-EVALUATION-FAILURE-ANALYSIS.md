# PHASE-07A - Evaluation And Failure Analysis

## Goal

Build comparable evaluation for checkpoints and avoid vibes-only assessment.

## Scope

- Loss, perplexity, throughput, memory, and generation metrics.
- Fixed qualitative prompt set.
- Toy task probes.
- Failure taxonomy and report generation.

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

- Add evaluation scripts that run against any compatible checkpoint.
- Add fixed prompts for English, Japanese, technical, instruction, and bilingual probes.
- Add report generation with failure taxonomy.
- Compare serious model scales in docs.

## Acceptance Criteria

- Evaluation can run against any compatible checkpoint.
- Each serious model scale has a comparable report.
- Reports classify failures such as repetition, memorization, language mixing, and tokenization artifacts.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- Qualitative samples can mislead without fixed prompts and comparable settings.

## Out Of Scope

- External leaderboard optimization.
