# PHASE-06A - Instruction Tuning Layer

## Goal

Convert a pretrained scratch model into a simple instruction-following variant.

## Scope

- Small instruction dataset.
- Supervised fine-tuning loop.
- Prompt format and evaluation probes.
- Base-vs-SFT comparison.

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

- Define the instruction prompt format.
- Add SFT data loading and training.
- Add base-vs-SFT fixed prompt comparisons.
- Document what the model can and cannot follow.

## Acceptance Criteria

- Instruction-tuned checkpoint exists outside git.
- Base-vs-SFT comparison is documented.
- Narrow instruction-following behavior improves on fixed probes.
- Limitations are clear and not overstated.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- Small base models may not have enough language ability for robust instruction following.

## Out Of Scope

- Production safety claims.
- RLHF or preference optimization.
