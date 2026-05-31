# PHASE-03A Transformer Architecture Note

## Scope

PHASE-03A introduces the first scratch-owned decoder-only causal Transformer for the project. It is intentionally small enough for CPU smoke tests but uses the same contract later phases can scale: token batches enter as `[batch, time]`, the model returns logits as `[batch, time, vocab]`, and training targets are the same stream shifted by one token.

## Base Architecture

- Token embeddings: learned `nn.Embedding`.
- Position representation: learned absolute positional embeddings.
- Blocks: pre-norm Transformer blocks.
- Attention: multi-head causal self-attention with an explicit lower-triangular boolean mask.
- MLP: GELU feed-forward block.
- Residuals: residual connection around attention and MLP sublayers.
- Final head: final LayerNorm then a tied input/output language-model head.
- Loss: cross entropy over pre-shifted next-token targets from the PHASE-02A token batch sampler.

This matches the North Star's recommended first implementation: learned positions, LayerNorm, GELU, multi-head self-attention, tied embedding/head weights, and float32 training.

## Config And Device Policy

`configs/transformer_micro.yaml` defines the PHASE-03A micro model and points at the PHASE-02A tokenized smoke dataset. The model config records vocabulary size, context length, embedding dimension, layers, heads, MLP size, dropout, and whether embeddings are tied.

Device support is deliberately conservative:

- `cpu` always runs.
- `mps` is accepted, but falls back to CPU when unavailable.
- `dtype` must be `float32` until a later phase justifies mixed precision.

## Smoke Training And Artifacts

`python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20` regenerates the ignored PHASE-02A tokenized smoke data if needed, trains on a fixed micro batch, logs `metrics.jsonl`, writes `manifest.json`, and saves `checkpoint_last.pt` under the ignored run directory.

The fixed batch is intentional for PHASE-03A: the goal is to prove model wiring, causal loss flow, optimizer behavior, checkpoint metadata, and generation compatibility before larger PHASE-04A pretraining.

## Generation

`python -m inference.generate --config configs/transformer_micro.yaml --prompt hello --max-new-tokens 16 --seed 123` loads the smoke checkpoint by default and performs autoregressive decoding. Greedy decoding uses `temperature: 0.0`; positive temperatures use seeded sampling with optional top-k filtering.

## Deferred Decisions

- RoPE: deferred until longer-context experiments need better extrapolation.
- RMSNorm: deferred until there is comparative evidence against LayerNorm.
- SwiGLU: deferred until a later scale phase can compare parameter and throughput tradeoffs.
- GQA: deferred until larger models or inference memory pressure make it useful.
- KV cache: deferred to PHASE-08A Mac-native inference unless an earlier benchmark requires it.
- Gradient checkpointing: deferred until model depth or memory pressure justifies the added complexity.
- Mixed precision: deferred until the dtype policy can be tested across CPU, MPS, and later MLX paths.
