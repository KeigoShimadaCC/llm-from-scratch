# PHASE-05A Scaling Report

## Summary

- 30M gate status: pass: kgpt-30m trained locally from random initialization to the configured 40-step PHASE-05A completion point.
- Final tokenizer: phase02a-byte-bpe-bilingual
- Data mixture: Approximately 39% English, 36% Japanese, and 25% mixed Japanese/English by record count.
- Profiling method: Final metrics.jsonl row from local CPU kgpt-30m smoke run.

## Run Comparison

| Run | Status | Parameters | Initial loss | Final loss | Improvement | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| phase03a_transformer_micro | trained | 29,728 | 22.2509 | 1.0930 | 95.09% | 1280 |
| phase04a_tiny_smoke | trained | 5,633,536 | 164.5951 | 4.0593 | 97.53% | 12800 |
| phase05a_kgpt30m_smoke | trained | 31,734,272 | 316.7530 | 11.4647 | 96.38% | 1280 |

## Tokenizer Decision

Use the repo-owned byte-level BPE tokenizer because it has completed provenance, byte fallback, English/Japanese roundtrip behavior, and successful PHASE-04A tiny-pretraining evidence.

## Data Mixture

- Source manifest: `docs/phase05a_data_mixture_manifest.json`
- English: 14/36 records
- Japanese: 13/36 records
- Mixed English/Japanese: 9/36 records
- License: Repo-authored fixture for this project; no third-party text source.

## Mac Profiling

- Device: cpu
- Dtype: float32
- Context length: 32
- Batch size: 1
- Tokens/sec: 254.04497570222955
- Peak memory: not available from CPU metrics helper

## Bottlenecks

- The corpus is too small for meaningful language quality claims.
- CPU throughput is sufficient for smoke evidence but not long 30M+ pretraining.
- Japanese byte-level segmentation remains inefficient under the provisional tokenizer.
- Memory profiling on CPU lacks peak memory; MPS/MLX profiling is deferred to later inference and optimization phases.

## Stretch Decisions

- kgpt-50m dry-run passed with 59,345,280 parameters; actual training deferred until the 30M run is reviewed.
- kgpt-100m dry-run passed with 113,721,600 parameters; actual training deferred until 50M evidence and a longer run budget exist.

## Phase Gate

The North Star 30M+ requirement is satisfied for local educational scale-gate evidence. The run is not a quality claim because the corpus is intentionally tiny.
