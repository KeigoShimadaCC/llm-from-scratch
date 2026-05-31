# Structure And Tech Specs

## Stack

- Python 3.11 or 3.12.
- `uv` for environment and command execution.
- PyTorch first, with MPS support where practical.
- MLX later for Mac-native inference and optimization.
- `pytest` for tests and `ruff` for linting.
- Config-driven experiments using YAML or JSON.

## Initial Repo Shape

```text
configs/
data/
docs/
kgpt/
tokenizer/
train/
eval/
inference/
experiments/
tests/
```

## Core Architecture

The scratch model path targets a decoder-only causal Transformer:

- token embeddings;
- positional representation;
- pre-norm Transformer blocks;
- multi-head causal self-attention;
- MLP block;
- residual connections;
- final norm;
- tied language-model head when useful;
- autoregressive generation.

## Validation

Global runner validation starts conservative and should be tightened as PHASE-01A creates the project skeleton:

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

Phase plans may add more specific experiment checks.

## Constraints

- Keep private datasets, processed data, checkpoints, and local experiment runs out of git.
- Every scale-up must follow a passing tiny overfit test.
- Tokenizer quality must be reported before serious token-level training.
- Separate scratch-core work from practical reference/open-weight comparison work.
