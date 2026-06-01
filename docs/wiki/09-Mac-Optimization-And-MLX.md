# 09. Mac Optimization And MLX

## Goal

Understand what optimization exists today and what remains deferred.

## Why It Matters

Apple Silicon changes practical engineering decisions. CPU and PyTorch MPS can run the educational path today. KV-cache
parity is tested for inference behavior. MLX is the likely future Mac-native optimization path, but tensor-loading
parity is not implemented yet.

## Repo Map

- [Inference benchmark report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_benchmark.md)
- [KV-cache parity output](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_kv_cache_parity.json)
- [MLX deferral](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_mlx_deferral.md)
- [Inference benchmark CLI](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/inference/benchmark.py)

## Run It

```bash
uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml
uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 --output docs/phase08a_benchmark.md
```

## Inspect It

If MLX is marked deferred in the benchmark report, that is expected. The repo should not claim PyTorch-vs-MLX parity
until an MLX model-loading path and logits parity tests exist.

## Try Changing

Run the benchmark on a machine with MPS available and compare CPU vs MPS rows. Keep any benchmark artifacts as compact
reports, not raw run directories.

## Further Reading

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX documentation](https://mlx-framework.org/)
