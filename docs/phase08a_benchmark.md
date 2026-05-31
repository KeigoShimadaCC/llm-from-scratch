# PHASE-08A Inference Benchmark

## Protocol

- Config: `configs/inference_benchmark.yaml`
- Checkpoint: `experiments/runs/phase03a_transformer_micro/checkpoint_last.pt`
- Model config: `configs/transformer_micro.yaml`
- Prompt count: 3
- Warmup runs: 1
- Measured runs: 2
- Default max new tokens: 32
- Sampling: greedy for benchmark stability.

## Prompts

- `hello`
- `The model learns`
- `Q:say hi
A:`

## Results

| Device | Status | Uncached tok/s | Cached tok/s | Cached latency sec | Memory | Reason |
|---|---:|---:|---:|---:|---|---|
| cpu | measured | 2256.9578 | 2498.0152 | 0.0129 | {"device": "cpu"} |  |
| mps | measured | 612.9689 | 572.3221 | 0.0605 | {"mps_current_allocated_bytes": 121088} |  |
| mlx | deferred | n/a | n/a | n/a | {} | MLX is not installed; see docs/phase08a_mlx_deferral.md. |

## MLX Status

MLX is compared when an importable `mlx` runtime and a compatible scratch-model loading path are available. If the
table marks MLX as deferred, see `docs/phase08a_mlx_deferral.md` for blocker evidence and follow-up.
