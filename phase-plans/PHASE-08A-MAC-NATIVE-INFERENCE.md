# PHASE-08A - Mac-Native Inference

## Goal

Make the trained scratch model pleasant to run locally on Apple Silicon.

## Scope

- PyTorch MPS inference baseline.
- MLX inference path or export if practical.
- KV cache for faster generation.
- CLI for completion/chat and inference benchmarks.

## Prerequisites

- At least one trained checkpoint is available from prior phases.
- Model config, tokenizer, checkpoint metadata, and sampling settings are compatible with inference loading.
- Evaluation prompts and expected generation behavior are available for parity tests.

## Phase Dependencies

- Depends on PHASE-05A or PHASE-06A checkpoints plus PHASE-07A fixed prompts for parity and benchmark evidence.
- Unblocks PHASE-09A by producing Mac-local inference, benchmark, and MLX comparison evidence.

## Allowed Paths

- `configs/**`
- `docs/**`
- `kgpt/**`
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

- Add `generate`, `chat`, and benchmark commands with documented CLI options.
- Support CLI options for checkpoint path, config path, tokenizer path, prompt, max new tokens, temperature, top-k, top-p, seed, device, dtype, stop token/string, repetition penalty when implemented, and output path.
- Implement greedy, temperature, top-k, and top-p sampling with deterministic seeded behavior.
- Define stop-token behavior and max-new-token behavior.
- Add KV-cache support if not already present.
- Add KV-cache parity tests showing cached and uncached generation agree where deterministic.
- Add PyTorch MPS benchmark.
- Add MLX inference path or produce a formal deferral gate with blocker evidence, human or unattended-policy approval, and a concrete follow-up plan. Default expectation is PyTorch-vs-MLX comparison.
- Add CPU, MPS, and MLX benchmark protocol with prompt set, model, context length, max tokens, warmup, measured runs, tokens/sec, latency, and memory where measurable.
- Write a model card or local inference guide.

## Deliverables

- Completion CLI.
- Chat/instruction-format CLI.
- Inference benchmark CLI.
- KV-cache implementation or documented deferral if not compatible.
- PyTorch MPS and MLX comparison report, or formal MLX deferral.
- Model card or local inference guide.

## Evidence Artifacts

- Benchmark report for CPU, MPS, and MLX when available.
- Parity test output for cached vs uncached generation.
- Example outputs from fixed prompts.
- Model loading compatibility report covering config, tokenizer, and checkpoint metadata.

## Artifact Policy

- Benchmark logs and long generated output files may remain ignored if summarized.
- Commit CLI code, small tests, model card, and benchmark summary. Do not commit checkpoints.

## Acceptance Criteria

- A trained model can run locally without cloud services.
- Latency and throughput are measured.
- PyTorch MPS and MLX paths are compared, or MLX is formally deferred with blocker evidence and human or unattended-policy approval.
- CLI usage is documented.
- Sampling algorithms, stop behavior, and deterministic seeding are tested or documented.
- KV-cache output parity is tested when KV cache exists.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m inference.generate --config configs/inference_smoke.yaml --prompt hello --max-new-tokens 16 --seed 123`
- `uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml`
- `uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 --output docs/phase08a_benchmark.md`

## Human Decisions

- Approve formal MLX deferral if MLX cannot be completed in this phase.
- Approve model/checkpoint used for the final inference guide and benchmarks.

## Phase Gate

Mark complete only when local inference works without cloud services, benchmark evidence exists for available devices, and PyTorch-vs-MLX comparison or formal MLX deferral is documented.

## Risks

- MLX integration may require a separate compatibility layer.

## Out Of Scope

- Production serving infrastructure.
- Hosted web app deployment.

## Deferred Backlog

- Optional local web UI remains deferred unless the core CLI and benchmark requirements are already satisfied.
