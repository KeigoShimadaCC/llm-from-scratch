# Command Index

This index lists the commands a fresh engineer should use to rebuild the main evidence path. Commands that train or
generate local artifacts write to ignored directories such as `experiments/runs/` or `data/tokenized/`.

## Setup And Baseline Validation

```bash
uv sync
uv run pytest
uv run ruff check .
git diff --check
```

## PHASE-01A Micro Character Loop

```bash
uv run python -m train.micro_char --config configs/micro_char.yaml --max-steps 200 --run-name phase01a_overfit_smoke
uv run python -m inference.generate_char --checkpoint experiments/runs/phase01a_overfit_smoke/checkpoint_last.pt --prompt hello --seed 123 --max-new-tokens 32
```

## PHASE-02A Tokenizer And Data Pipeline

```bash
uv run python -m tokenizer.train_report --config configs/tokenizer_bilingual.yaml --output docs/tokenizer_report.md
uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2
```

## PHASE-03A Transformer Smoke

```bash
uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20
uv run python -m inference.generate --config configs/transformer_micro.yaml --prompt hello --max-new-tokens 16 --seed 123
```

## PHASE-04A Tiny Pretraining

```bash
uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke
uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --checkpoint experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt --output docs/phase04a_tiny_report.md
```

## PHASE-05A 30M Scale Gate

The 30M command is a local scale-gate run. It should be reviewed for machine budget before increasing steps or data.

```bash
uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 --run-name phase05a_kgpt30m_smoke
uv run python -m train.pretrain --config configs/kgpt_50m.yaml --dry-run
uv run python -m train.pretrain --config configs/kgpt_100m.yaml --dry-run
uv run python -m eval.compare_runs --manifest docs/phase05a_scaling_manifest.json --output docs/phase05a_scaling_report.md
```

## PHASE-06A Instruction Tuning

```bash
uv run python -m train.sft --config configs/sft_smoke.yaml --max-steps 50 --run-name phase06a_sft_smoke
uv run python -m eval.sft_compare --config configs/sft_eval.yaml --output docs/phase06a_sft_eval.md
```

## PHASE-07A Evaluation

```bash
uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --output docs/phase07a_eval_report.md
uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest.json --output docs/phase07a_checkpoint_comparison.md
```

## PHASE-08A Inference And Benchmarks

```bash
uv run python -m inference.generate --config configs/inference_smoke.yaml --prompt hello --max-new-tokens 16 --seed 123
uv run python -m inference.chat --config configs/inference_smoke.yaml --instruction "say hi" --max-new-tokens 16 --seed 123
uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml
uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 --output docs/phase08a_benchmark.md
```

## PHASE-09A Final Audits

```bash
uv run python -m eval.audit_claims --doc docs/FINAL_WRITEUP.md --output docs/claim_evidence_audit.md
uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md
```

## PHASE-10A Corpus Source Registry

```bash
uv run python -m corpus.audit_sources --config configs/corpus_v01.yaml --output docs/corpus_v01_source_manifest.md
```

## PHASE-10B Corpus Download And Cleaning

```bash
uv run python -m corpus.download --config configs/corpus_v01.yaml --dry-run
uv run python -m corpus.clean --config configs/corpus_v01.yaml --smoke --output data/processed/corpus_v01_smoke
```

## PHASE-10C Split, Leakage, And Dataset Manifest

```bash
uv run python -m corpus.split_manifest --config configs/corpus_v01.yaml --processed data/processed/corpus_v01_smoke --output docs/corpus_v01_dataset_manifest.json
```

## PHASE-10D Tokenizer And Tokenized Dataset

```bash
uv run python -m tokenizer.train_report --config configs/tokenizer_corpus_v01.yaml --output docs/tokenizer_corpus_v01_report.md
uv run python -m train.sample_batches --config configs/corpus_v01_tokenized.yaml --max-batches 2
```

## PHASE-11A Real-Corpus 30M Training

The PHASE-11A training command is a local 30M smoke run over ignored `corpus_v01` token arrays. Review local
runtime and thermals before increasing step count, context length, or corpus size.

```bash
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --dry-run --validate-resume
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --max-steps 1000 --run-name phase11a_kgpt30m_corpus_v01_smoke
uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest_corpus_v01.json --output docs/phase11a_real_corpus_checkpoint_comparison.md
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "The model learns" --max-new-tokens 16 --seed 123
```

## Current Limitations For Reproduction

- Checkpoints and generated run directories are ignored. A fresh clone can rerun commands to regenerate them locally.
- The 30M command can be expensive relative to the tiny fixture data. Keep the phase config unchanged when reproducing
  the committed scale-gate evidence.
- MLX commands are absent because MLX loading and parity are formally deferred.
