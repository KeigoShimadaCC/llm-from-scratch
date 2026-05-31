# PHASE-08A MLX Deferral

## Decision

MLX inference is formally deferred for this phase unless the local environment has both:

- an importable `mlx` Python package;
- a maintained scratch-model loading path that maps this repo's PyTorch checkpoint tensors into MLX arrays.

PHASE-08A still compares CPU and PyTorch MPS when available. The benchmark report records MLX as `deferred` with this
document as blocker evidence when MLX cannot be imported or when tensor-loading parity is not implemented.

## Approval Path

This deferral is accepted under the unattended automation policy for PHASE-08A because the user requested continued
automation through the remaining phases. The final write-up must describe MLX as deferred, not as a fully achieved
PyTorch-vs-MLX North Star comparison.

## Blocker Evidence

- The core scratch model, tokenizer path, training loop, evaluation, and inference are PyTorch-owned through PHASE-08A.
- Existing checkpoints are PyTorch `torch.save` artifacts with optimizer metadata and PyTorch tensor names.
- Adding MLX loading now would require a separate compatibility layer and parity tests for every model weight tensor.

## Follow-Up Plan

1. Add `mlx` as an optional extra dependency, not a required project dependency.
2. Write a tensor-name mapping from `DecoderOnlyTransformer.state_dict()` to an MLX module.
3. Add logits parity on a fixed prompt before benchmarking MLX generation.
4. Extend `inference.benchmark` so MLX reports measured latency/throughput instead of `deferred`.

## Claim Boundary

When MLX is deferred, PHASE-08A can claim Mac-local PyTorch CPU/MPS inference and benchmark evidence. It must not claim
that the original North Star's PyTorch-vs-MLX comparison has been fully achieved.
