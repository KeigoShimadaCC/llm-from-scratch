# PHASE-01A - MicroGPT Character LM

## Goal

Build the smallest complete language-modeling loop using character-level data.

## Scope

- Character vocabulary and encode/decode utilities.
- Tiny baseline and/or tiny Transformer path.
- Next-character prediction.
- Greedy and sampling-based generation.
- Validation loss and a tiny overfit test.
- Short explanation of underfitting, overfitting, and sampling temperature.

## Allowed Paths

- `configs/**`
- `data/README.md`
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

- Add character tokenizer utilities and tests.
- Add a small public fixture corpus for tests only.
- Add a micro training path that can overfit tiny text.
- Add sample generation with deterministic seeding and temperature.
- Document the first training curve and sample behavior.

## Acceptance Criteria

- Character tokenizer roundtrip tests pass.
- Training loss decreases on the tiny corpus.
- The model can overfit a tiny dataset.
- Generation produces non-empty text and supports deterministic seeds.
- A short experiment note exists under `docs/`.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- A model that cannot overfit tiny text is not ready for token-level work.

## Out Of Scope

- BPE or SentencePiece tokenizer work.
- 5M+ parameter training.
- Instruction tuning.
