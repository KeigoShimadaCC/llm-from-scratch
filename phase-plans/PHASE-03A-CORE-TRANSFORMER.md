# PHASE-03A - Core Decoder-Only Transformer

## Goal

Implement the real decoder-only causal Transformer architecture in the scratch-core path.

## Scope

- Token and positional embeddings.
- Pre-norm Transformer blocks.
- Multi-head causal self-attention.
- MLP, residuals, final norm, and language-model head.
- Autoregressive generation.
- Parameter counting and shape/mask tests.

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

- Implement attention, blocks, model config, and sampling interfaces.
- Add tests for causal masking, logits shape, parameter count, and generation.
- Integrate model creation with configs and the training loop.
- Document first architecture choices and deferred upgrades.

## Acceptance Criteria

- Attention mask prevents future-token access.
- Logits shape is `[batch, time, vocab]`.
- Parameter count is computed programmatically.
- The model trains stably at micro scale.
- Generation is autoregressive and causal.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- Shape and mask bugs can make loss curves misleading.

## Out Of Scope

- RoPE, RMSNorm, SwiGLU, GQA, KV cache, and gradient checkpointing unless needed for a narrow test.
- Long pretraining.
