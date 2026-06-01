# Appendix. Command Index

This page collects the most useful commands for the wiki lessons. The canonical source remains the repo's
[Command Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/COMMAND_INDEX.md).

## Setup

```bash
uv sync
uv run pytest
uv run ruff check .
git diff --check
```

## Tokenizer And Data

```bash
uv run python -m corpus.audit_sources --config configs/corpus_v01.yaml --output docs/corpus_v01_source_manifest.md
uv run python -m corpus.clean --config configs/corpus_v01.yaml --smoke --output data/processed/corpus_v01_smoke
uv run python -m corpus.split_manifest --config configs/corpus_v01.yaml --processed data/processed/corpus_v01_smoke --output docs/corpus_v01_dataset_manifest.json
uv run python -m tokenizer.train_report --config configs/tokenizer_corpus_v01.yaml --output docs/tokenizer_corpus_v01_report.md
uv run python -m train.sample_batches --config configs/corpus_v01_tokenized.yaml --max-batches 2
```

## Training And Evaluation

```bash
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --dry-run --validate-resume
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --max-steps 1000 --run-name phase11a_kgpt30m_corpus_v01_smoke
uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest_corpus_v01.json --output docs/phase11a_real_corpus_checkpoint_comparison.md
```

## Inference

```bash
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "The model learns" --max-new-tokens 16 --seed 123
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "A local language model" --max-new-tokens 24 --temperature 0.8 --top-k 20 --seed 123
```

## Mac Inference Checks

```bash
uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml
uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 --output docs/phase08a_benchmark.md
```

## Further Reading

- [Command Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/COMMAND_INDEX.md)
- [Artifact Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/ARTIFACT_INDEX.md)
