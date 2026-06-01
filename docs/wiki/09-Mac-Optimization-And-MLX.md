# 09. Mac Optimization And MLX

## Goal

Understand what optimization exists today and what remains deferred.

## Why It Matters

Apple Silicon changes practical engineering decisions. CPU and PyTorch MPS can run the educational path today. KV-cache
parity is tested for inference behavior. MLX is the likely future Mac-native optimization path, but tensor-loading
parity is not implemented yet.

## What This Part Does

This lesson separates correctness from speed. First, the repo checks that cached inference produces the same behavior
as uncached inference. Then the benchmark compares CPU and MPS throughput. MLX is documented as deferred until there
is a scratch-model loading path and logits parity test, so the wiki should not imply MLX is already complete.

## Repo Map

- [Inference benchmark report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_benchmark.md):
  CPU/MPS throughput and MLX status.
- [KV-cache parity output](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_kv_cache_parity.json):
  cached-vs-uncached behavior evidence.
- [MLX deferral](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_mlx_deferral.md): blocker
  record for future MLX work.
- [Inference benchmark CLI](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/inference/benchmark.py):
  benchmark implementation.

## Run It

```bash
uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml
uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 --output docs/phase08a_benchmark.md
```

## Inspect It

If MLX is marked deferred in the benchmark report, that is expected. The repo should not claim PyTorch-vs-MLX parity
until an MLX model-loading path and logits parity tests exist.

Example benchmark row:

```text
cpu measured: cached tok/s 2811.4619
mps measured: cached tok/s 313.4925
mlx deferred: MLX is not installed
```

Those numbers came from a micro benchmark, not a universal performance claim. Rerun locally on your Mac before making
device decisions.

## Try Changing

Run the benchmark on a machine with MPS available and compare CPU vs MPS rows. Keep any benchmark artifacts as compact
reports, not raw run directories.

## Further Reading

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX documentation](https://mlx-framework.org/)
