# PHASE-08A Local Inference Guide

## Model

- Default config: `configs/inference_smoke.yaml`
- Default checkpoint: `experiments/runs/phase03a_transformer_micro/checkpoint_last.pt`
- Bootstrap behavior: if the default checkpoint is missing, the inference config regenerates the PHASE-03A micro
  Transformer smoke checkpoint as an ignored local artifact.

This guide uses the micro checkpoint for fast local validation. Larger tiny, 30M, or SFT checkpoints can be used by
changing the inference config once their ignored artifacts exist locally.

## Completion

```bash
uv run python -m inference.generate \
  --config configs/inference_smoke.yaml \
  --prompt hello \
  --max-new-tokens 16 \
  --seed 123
```

Important options:

- `--checkpoint`: override checkpoint path.
- `--tokenizer`: override tokenizer JSON path.
- `--temperature 0`: greedy decoding.
- `--top-k 0`: disable top-k filtering.
- `--top-p 0`: disable nucleus filtering.
- `--repetition-penalty`: apply a penalty of at least `1.0` to seen token logits.
- `--stop-string`: truncate decoded output after a stop string.
- `--stop-token-id`: stop generation when the sampled token id appears.
- `--device cpu|mps`: request CPU or PyTorch MPS. MPS falls back through the shared device resolver when unavailable.
- `--use-cache` / `--no-cache`: select cached or uncached generation.
- `--output`: write the JSON payload to a file.

## Chat

```bash
uv run python -m inference.chat \
  --config configs/inference_smoke.yaml \
  --instruction "say hi" \
  --max-new-tokens 16 \
  --seed 123
```

The PHASE-08A chat path is a compact instruction formatter, not a full multi-turn chat system.

## KV-Cache Parity

```bash
uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml
```

The parity check uses deterministic greedy generation and fails if cached and uncached decoded outputs differ.

## Benchmark

```bash
uv run python -m inference.benchmark \
  --config configs/inference_benchmark.yaml \
  --max-new-tokens 32 \
  --output docs/phase08a_benchmark.md
```

The benchmark measures CPU and PyTorch MPS when available. MLX is recorded as deferred unless the runtime and model
loading compatibility layer are available.

## Evidence Files

- Model loading compatibility: `docs/phase08a_model_loading_report.md`
- KV-cache parity output: `docs/phase08a_kv_cache_parity.json`
- Generation example: `docs/phase08a_generation_example.json`
- Benchmark summary: `docs/phase08a_benchmark.md`
- MLX deferral: `docs/phase08a_mlx_deferral.md`

## Current Limitations

- The default checkpoint is a tiny smoke model, so output quality is not a language-quality claim.
- MLX parity is deferred in `docs/phase08a_mlx_deferral.md`.
- Stop strings are applied as decoded-output truncation after token generation; stop token ids stop during sampling.
