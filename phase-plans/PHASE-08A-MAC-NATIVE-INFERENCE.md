# PHASE-08A - Mac-Native Inference

## Goal

Make the trained scratch model pleasant to run locally on Apple Silicon.

## Scope

- PyTorch MPS inference baseline.
- MLX inference path or export if practical.
- KV cache for faster generation.
- CLI for completion/chat and inference benchmarks.

## Allowed Paths

- `configs/**`
- `docs/**`
- `kgpt/**`
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

- Add `generate`, `chat`, and benchmark commands.
- Add KV-cache support if not already present.
- Add PyTorch MPS benchmark.
- Add MLX path or clearly document why it is deferred.
- Write a model card or local inference guide.

## Acceptance Criteria

- A trained model can run locally without cloud services.
- Latency and throughput are measured.
- PyTorch MPS and MLX paths are compared when both exist.
- CLI usage is documented.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- MLX integration may require a separate compatibility layer.

## Out Of Scope

- Production serving infrastructure.
- Hosted web app deployment.
