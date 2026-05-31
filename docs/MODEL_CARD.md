# Model Card

## Model Family

KeigoGPT-Lab is a scratch-owned educational decoder-only language-model family. The repo implements tokenizer
training/configuration, token batching, model architecture, training loops, evaluation, and local inference without
using Hugging Face `AutoModelForCausalLM` or equivalent pretrained wrappers in the core path.

## Variants

| Variant | Parameters | Status | Evidence |
|---|---:|---|---|
| PHASE-01A character LM | small debug model | trained overfit smoke | `docs/phase01a_micro_char_report.md` |
| PHASE-03A micro Transformer | 29,728 | trained smoke | `docs/phase03a_transformer_architecture.md`, `docs/phase07a_checkpoint_comparison.md` |
| PHASE-04A `kgpt-tiny-5m` | 5,633,536 | trained tiny pretraining | `docs/phase04a_tiny_report.md` |
| PHASE-05A `kgpt-30m` | 31,734,272 | trained 30M+ scale gate | `docs/phase05a_scaling_report.md` |
| PHASE-05A `kgpt-50m` | 59,345,280 | dry-run only | `docs/phase05a_scaling_report.md` |
| PHASE-05A `kgpt-100m` | 113,721,600 | dry-run only | `docs/phase05a_scaling_report.md` |
| PHASE-06A SFT variant | 31,734,272 | trained narrow SFT smoke | `docs/phase06a_sft_eval.md` |

## Intended Use

- Educational inspection of an end-to-end local LLM stack.
- Reproducible smoke experiments for tokenizer, training, evaluation, and inference code.
- Small-scale Mac-local benchmarking and debugging.

## Out-Of-Scope Use

- General factual question answering.
- Production assistant use.
- Safety-critical advice.
- Benchmark claims against public LLMs.
- Claims of robust Japanese or bilingual fluency.

## Training Data

The committed evidence uses repo-authored fixture data only. See `docs/DATA_CARD.md` for source, license, split,
deduplication, leakage, and contamination notes.

## Evaluation Summary

- Tiny pretraining improved validation loss by 97.53% on fixture-scale data.
- The 30M+ run completed the configured local scale gate but remains undersampled.
- SFT improved fixed command probes but regressed held-out SFT validation.
- Evaluation reports distinguish live checkpoint rows from summary-only rows when ignored checkpoints are absent.

## Limitations

The models are not useful general-purpose language models. Current evidence is about implementation correctness,
reproducibility, local training mechanics, and failure analysis. Output may repeat, ignore instructions, contain
tokenization artifacts, or memorize tiny fixture patterns.
