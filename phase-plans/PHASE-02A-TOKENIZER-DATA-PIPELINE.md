# PHASE-02A - Tokenizer And Dataset Pipeline

## Goal

Move from character-level modeling to real token-level LLM training.

## Scope

- Train or configure a small tokenizer for English/Japanese experiments.
- Compare English-only and bilingual Japanese/English tokenization.
- Implement train/validation split generation.
- Add chunked or memory-mapped token files.
- Add batch sampling for language-model training.
- Report tokenizer statistics and failure cases.

## Allowed Paths

- `configs/**`
- `data/README.md`
- `docs/**`
- `kgpt/**`
- `tokenizer/**`
- `train/**`
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

- Add tokenizer training or configuration scripts.
- Add tokenizer inspection and report generation.
- Add dataset preprocessing and batch sampling.
- Add roundtrip, split-integrity, and batch-shape tests.
- Document English/Japanese tokenization tradeoffs.

## Acceptance Criteria

- Tokenizer encode/decode roundtrip tests pass.
- Dataset batching returns expected tensor shapes.
- Tokenizer report includes vocabulary size, compression, and bad segmentation examples.
- A small model can consume token batches.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- Large or private datasets must remain outside git.
- Oversized vocabularies can waste parameters in small models.

## Out Of Scope

- Long pretraining runs.
- Final main-model tokenizer decision if evidence is insufficient.
