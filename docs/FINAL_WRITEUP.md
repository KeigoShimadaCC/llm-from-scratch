# Final Write-Up

## Final Positioning

Status: North Star partially achieved with documented fallback.

The repo now contains a scratch-owned path from text fixtures through tokenization, token batches, a decoder-only
Transformer, training, evaluation, instruction tuning, and local inference. The evidence supports an educational
research-engineering claim: the system trains from random initialization, tracks loss and samples, reaches a local
30M+ scale gate, and exposes a reproducible Mac inference CLI.

It does not support a claim that the original North Star is fully complete. MLX inference is formally deferred, the
50M and 100M stretch models were dry-run/profile targets rather than trained checkpoints, and the instruction-tuned
checkpoint improved only narrow fixed probes while held-out SFT validation regressed. Those outcomes are kept as
evidence-backed limitations rather than hidden gaps.

## Architecture

PHASE-03A introduced the core scratch model in `kgpt/transformer.py`: learned token embeddings, learned absolute
positions, pre-norm Transformer blocks, causal multi-head self-attention, GELU MLPs, residual connections, final
LayerNorm, tied input/output embeddings, shifted next-token loss, and autoregressive generation. PHASE-08A added
opt-in KV-cache generation while preserving uncached parity checks.

The first implementation intentionally stayed conservative: float32, PyTorch CPU/MPS, LayerNorm, GELU, and learned
positions. RoPE, RMSNorm, SwiGLU, GQA, mixed precision, and full MLX parity remain future optimization work.

## Tokenizer And Data

PHASE-01A used a character tokenizer to prove the smallest overfit loop. PHASE-02A moved to a repo-owned UTF-8
byte-level BPE tokenizer with byte fallback, deterministic roundtrip behavior, explicit English/Japanese stats,
and a manifest-backed tokenized dataset format.

All committed data evidence is repo-authored fixture data. The manifests document source names, licenses, split
methods, deduplication, leakage checks, token file hashes, and language mix. Private data, raw corpora, processed
corpora, tokenized arrays, and checkpoints are intentionally excluded from git.

## Training Dynamics

The training path moved through three useful scales:

| Scale | Evidence | Result |
|---|---|---|
| Micro character/model loops | `docs/phase01a_micro_char_report.md`, `docs/phase03a_transformer_architecture.md` | Wiring, determinism, overfit behavior, causal loss, and generation were proven. |
| Tiny pretraining | `docs/phase04a_tiny_report.md` | `kgpt-tiny-5m` improved validation loss by 97.53% over 200 local steps. |
| 30M+ scale gate | `docs/phase05a_scaling_report.md` | `kgpt-30m` trained locally for the configured 40-step completion run with 31,734,272 parameters. |

These runs are not language-quality claims. The corpora are intentionally tiny, so loss reductions mostly prove the
training stack, checkpointing, and scale gates.

## Scaling

The 30M+ requirement is satisfied at the educational scale-gate level by `kgpt-30m`. The 50M and 100M configs were
validated as dry-run/profile targets but not trained. The practical bottlenecks were CPU throughput, tiny data,
Japanese byte segmentation inefficiency, and missing long-run budget.

## Instruction Tuning

PHASE-06A added a compact instruction fixture, prompt-template versioning, train/validation splits, response-only
loss masking, SFT training, and base-vs-SFT comparison. The SFT model exactly reproduced the narrow command probes
in the committed report, but held-out SFT validation loss regressed from 32.4693 to 63.1710. The valid claim is
therefore limited to a narrow instruction-smoke path, not general alignment or assistant behavior.

## Evaluation And Failure Analysis

PHASE-07A consolidated fixed prompts and one comparison schema across micro, tiny, 30M+, and SFT checkpoints. Fresh
clones can regenerate live evidence when ignored checkpoints exist; otherwise committed reports mark rows as
summary-only. The failure taxonomy calls out repetition loops, bad token boundaries, instruction ignoring, tiny-corpus
memorization risk, undersampled 30M runs, and held-out SFT regression.

## Mac Inference

PHASE-08A added completion and compact chat CLIs with checkpoint/config/tokenizer overrides, greedy and sampled
generation, top-k, top-p, repetition penalty, stop strings, stop token ids, CPU/MPS device selection, optional KV-cache
generation, parity checks, and a CPU/MPS benchmark protocol. Local PyTorch CPU and MPS inference are measured. MLX is
documented as deferred until an optional dependency, tensor mapping, and logits parity layer exist.

## What Changed Between Phases

| Phase | Change |
|---|---|
| PHASE-00B | Established Python package metadata, config loading, deterministic seeding, checkpoint helpers, dummy training, and validation. |
| PHASE-01A | Proved a tiny character-level overfit loop and deterministic generation. |
| PHASE-02A | Added bilingual byte-level BPE, data manifests, dedup/split/leakage checks, token files, and batch sampling. |
| PHASE-03A | Added the scratch decoder-only Transformer and smoke training/generation path. |
| PHASE-04A | Added tiny pretraining with validation curves, checkpoint resume behavior, and fixed prompt samples. |
| PHASE-05A | Added 30M+ scaling config, local 30M run evidence, stretch dry runs, and profiling notes. |
| PHASE-06A | Added instruction tuning, prompt template versioning, response-only loss masking, and base-vs-SFT comparison. |
| PHASE-07A | Added cross-checkpoint evaluation, failure taxonomy, leakage/memorization probes, and comparison reports. |
| PHASE-08A | Added local inference CLIs, sampling controls, KV-cache parity, CPU/MPS benchmark, and MLX deferral evidence. |
| PHASE-09A | Added this evidence-backed final write-up, artifact index, command index, model card, data card, and claim audits. |

## Limitations

- The data is small and fixture-based, so the trained models are not useful general-purpose language models.
- Japanese tokenization still fragments under tiny byte-level BPE data.
- The 30M+ run is a local scale-gate run, not a long pretraining run.
- The SFT run demonstrates response-only training mechanics and narrow probe memorization, not robust instruction following.
- MLX inference and PyTorch-vs-MLX parity are deferred.
- Larger data, longer training, mixed precision, and stronger evaluation are future work.

## Claim To Evidence

| Claim | Evidence | Status |
|---|---|---|
| The repo owns the core decoder-only Transformer implementation. | `kgpt/transformer.py`, `docs/phase03a_transformer_architecture.md` | Supported |
| The tokenizer and data pipeline are documented with provenance, split, dedup, byte fallback, and leakage evidence. | `docs/tokenizer_report.md`, `docs/phase02a_data_manifest.json`, `docs/DATA_CARD.md` | Supported |
| Training runs are config-driven and produce ignored reproducible artifacts rather than committed checkpoints. | `docs/ARTIFACT_INDEX.md`, `docs/phase04a_tiny_report.md`, `docs/phase05a_scaling_manifest.json` | Supported |
| A 5M-20M tiny model was trained from random initialization. | `docs/phase04a_tiny_report.md` | Supported |
| A 30M+ model was trained locally from random initialization for the configured phase gate. | `docs/phase05a_scaling_report.md`, `docs/phase05a_scaling_manifest.json` | Supported |
| The 50M and 100M stretch models were prepared as dry-run targets but not trained. | `docs/phase05a_scaling_report.md`, `docs/phase05a_scaling_manifest.json` | Partial / deferred |
| A small instruction-tuned variant was produced, but evidence is limited to narrow probes and held-out regression is documented. | `docs/phase06a_sft_eval.md`, `docs/phase06a_instruction_data_manifest.json` | Partial |
| Evaluation reports compare micro, tiny, 30M+, and SFT checkpoints under one schema. | `docs/phase07a_checkpoint_comparison.md`, `docs/phase07a_eval_report.md`, `docs/phase07a_eval_schema.md` | Supported |
| Local PyTorch CPU/MPS inference is implemented with sampling controls, KV-cache parity, and benchmark evidence. | `docs/phase08a_inference_guide.md`, `docs/phase08a_benchmark.md`, `docs/phase08a_kv_cache_parity.json` | Supported |
| MLX inference and PyTorch-vs-MLX parity are not complete. | `docs/phase08a_mlx_deferral.md`, `docs/phase08a_benchmark.md` | Deferred |
| The final North Star status is partial rather than fully achieved. | `docs/FINAL_WRITEUP.md`, `docs/phase08a_mlx_deferral.md`, `docs/phase05a_scaling_report.md` | Partial |

## Next Work

1. Replace fixture-scale data with a larger licensed bilingual corpus and regenerate tokenizer/data manifests.
2. Train 50M and 100M checkpoints only after the 30M run has a meaningful data budget and target wall-clock budget.
3. Add MLX model loading, logits parity, cached generation parity, and benchmark rows.
4. Expand evaluation beyond smoke probes to include stronger repetition, memorization, bilingual, and toy reasoning tests.
