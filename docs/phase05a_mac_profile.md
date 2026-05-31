# PHASE-05A Mac Profile

## Profiled Run

- Run: `experiments/runs/phase05a_kgpt30m_smoke/`
- Config: `configs/kgpt_30m.yaml`
- Model: `kgpt-30m`
- Parameters: 31,734,272
- Device: CPU
- Dtype: float32
- Context length: 32
- Batch size: 1
- Gradient accumulation: 1
- Steps: 40
- Tokens seen: 1,280

## Throughput

The final logged step reported 254.04497570222955 tokens/sec. This is enough for local scale-gate validation but not enough for long 30M+ pretraining on meaningful data.

## Memory

CPU peak memory is not available from the current metrics helper. The run completed without memory failure and produced both `checkpoint_last.pt` and `checkpoint_best.pt`. MPS memory logging remains available in the metrics schema when running on MPS.

## Bottlenecks

- The current corpus is far too small for meaningful 30M generalization.
- CPU training throughput makes long 30M+ runs expensive without MPS/MLX optimization.
- Byte-level Japanese segmentation increases effective sequence cost.
- Checkpoint files are large enough that all model weights must remain ignored.

## Stretch Decision

`kgpt-50m` and `kgpt-100m` configs dry-run successfully, but actual stretch training is deferred until the 30M result is reviewed and a longer data/run budget is approved.
