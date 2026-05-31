# PHASE-08A Model Loading Report

## Config Compatibility

- Inference config: `configs/inference_smoke.yaml`
- Benchmark config: `configs/inference_benchmark.yaml`
- Model config: `configs/transformer_micro.yaml`
- Model kind: `transformer_smoke`
- Default checkpoint: `experiments/runs/phase03a_transformer_micro/checkpoint_last.pt`
- Bootstrap policy: regenerate the PHASE-03A micro Transformer checkpoint as an ignored local artifact when missing.

## Tokenizer Compatibility

- Tokenizer path is inherited from `configs/transformer_micro.yaml`.
- The tokenizer still uses the PHASE-02A byte-level BPE path with fallback training from `configs/tokenizer_bilingual.yaml`.
- Generated tokenized data and tokenizer artifacts remain ignored under `data/tokenized/**`.

## Checkpoint Metadata

The PHASE-08A smoke command loaded the bootstrapped checkpoint and reported:

- Model name: `kgpt-decoder-transformer-micro`
- Training step: 20
- Device: `cpu` for the required generation smoke.
- Dtype: `float32`

## Device Compatibility

- CPU generation passed.
- PyTorch MPS benchmark ran on this local machine and is summarized in `docs/phase08a_benchmark.md`.
- MLX is formally deferred in `docs/phase08a_mlx_deferral.md`.

## Artifact Policy

The checkpoint, tokenizer model, tokenized arrays, and benchmark scratch files are ignored artifacts. The committed
evidence is limited to configs, source code, tests, and summary reports.
