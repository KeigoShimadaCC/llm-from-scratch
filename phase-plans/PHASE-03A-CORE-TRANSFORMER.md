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

## Prerequisites

- PHASE-02A is complete enough to provide token batches or a documented synthetic token fallback for architecture tests.
- Character/micro training already proves the training loop can reduce loss.
- Config schema supports model dimensions, vocab size, context length, device, dtype, and sampling settings.

## Phase Dependencies

- Depends on PHASE-02A token/data interfaces and PHASE-01A training-loop behavior.
- Unblocks PHASE-04A by providing the scratch Transformer, architecture tests, and generation path.

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

- Implement attention, blocks, model config, and sampling interfaces.
- Implement next-token loss shifting explicitly: inputs predict targets shifted by one position.
- Implement or explicitly defer tied input/output embeddings; default is weight tying when dimensions permit.
- Define device/dtype policy for CPU, MPS, and fallback behavior. Default training dtype is float32 until mixed precision is justified.
- Add tests for causal masking, logits shape, parameter count, loss shifting, tied embeddings when enabled, and generation.
- Add a behavioral causal-mask test that proves changing a future token cannot change earlier logits.
- Integrate model creation with configs and the training loop.
- Document first architecture choices and deferred upgrades.

## Deliverables

- Decoder-only Transformer implementation.
- Model config compatibility with existing training configs.
- Parameter-count utility.
- Sampling-compatible forward/generation path.
- Architecture note under `docs/`.

## Evidence Artifacts

- Test output showing mask, shape, loss-shift, generation, and checkpoint compatibility tests pass.
- Architecture note with base choices and deferred decision points for RoPE, RMSNorm, SwiGLU, GQA, KV cache, weight tying, and gradient checkpointing.
- Ignored micro run proving the Transformer trains stably at micro scale.

## Artifact Policy

- Micro checkpoints and generated samples may be produced locally but must remain ignored.
- Architecture notes and small configs should be committed.

## Acceptance Criteria

- Attention mask prevents future-token access.
- Logits shape is `[batch, time, vocab]`.
- Parameter count is computed programmatically.
- The model trains stably at micro scale.
- Generation is autoregressive and causal.
- Loss shifting is tested and documented.
- Device/dtype behavior is documented and has a CPU fallback.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- Add the exact Transformer micro-train smoke and generation command here once implemented, and require them before marking PHASE-03A complete.

## Human Decisions

- Confirm whether base architecture remains learned positional embeddings + LayerNorm + GELU for first serious runs.
- Approve any upgrade from the first implementation to RoPE, RMSNorm, SwiGLU, GQA, KV cache, or mixed precision.

## Phase Gate

Mark complete only when architecture invariants are tested, a micro-scale training run is stable, and deferred upgrade decisions are documented rather than silently mixed into the base model.

## Risks

- Shape and mask bugs can make loss curves misleading.

## Out Of Scope

- RoPE, RMSNorm, SwiGLU, GQA, KV cache, and gradient checkpointing unless needed for a narrow test.
- Long pretraining.

## Deferred Backlog

- KV cache is expected in PHASE-08A unless needed earlier for an inference smoke test.
- Architecture upgrades require separate evidence and should not be bundled into scale-up without review.
