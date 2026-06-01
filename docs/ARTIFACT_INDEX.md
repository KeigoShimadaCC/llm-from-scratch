# Artifact Index

This repository commits summaries, manifests, schemas, and reports. It does not commit private data, generated
corpora, tokenized arrays, checkpoints, or full run directories.

## Committed Reports And Manifests

| Area | Committed evidence |
|---|---|
| North Star | `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md`, `NORTH_STAR.md` |
| Micro character LM | `docs/phase01a_micro_char_report.md` |
| Tokenizer and data | `docs/tokenizer_report.md`, `docs/phase02a_data_manifest.json` |
| Transformer architecture | `docs/phase03a_transformer_architecture.md` |
| Tiny pretraining | `docs/phase04a_tiny_report.md`, `docs/phase04a_data_manifest.json` |
| 30M+ scaling | `docs/phase05a_scaling_report.md`, `docs/phase05a_scaling_manifest.json`, `docs/phase05a_data_mixture_manifest.json`, `docs/phase05a_mac_profile.md`, `docs/phase05a_tokenizer_decision.md` |
| Instruction tuning | `docs/phase06a_sft_eval.md`, `docs/phase06a_instruction_data_manifest.json` |
| Evaluation | `docs/checkpoint_manifest.json`, `docs/phase07a_eval_schema.md`, `docs/phase07a_eval_report.md`, `docs/phase07a_checkpoint_comparison.md` |
| Inference | `docs/phase08a_inference_guide.md`, `docs/phase08a_benchmark.md`, `docs/phase08a_generation_example.json`, `docs/phase08a_kv_cache_parity.json`, `docs/phase08a_model_loading_report.md`, `docs/phase08a_mlx_deferral.md` |
| Final portfolio | `docs/FINAL_WRITEUP.md`, `docs/COMMAND_INDEX.md`, `docs/ARTIFACT_INDEX.md`, `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`, `docs/claim_evidence_audit.md` |
| Corpus source registry | `docs/corpus_v01_source_manifest.md` |
| Corpus split and tokenization | `docs/corpus_v01_dataset_manifest.json`, `docs/tokenizer_corpus_v01_report.md`, `docs/corpus_v01_tokenized_manifest.json` |
| Real-corpus 30M training | `docs/checkpoint_manifest_corpus_v01.json`, `docs/phase11a_real_corpus_checkpoint_comparison.md` |

## Ignored Local Artifact Locations

| Path | Contents | Recreate with |
|---|---|---|
| `experiments/runs/<timestamp>_phase00b_smoke/` | Dummy foundation run artifacts. | `uv run python -m train.dummy --config configs/dummy.yaml --run-name phase00b_smoke` |
| `experiments/runs/phase01a_overfit_smoke/` | Character LM metrics, checkpoint, samples, and manifest. | `uv run python -m train.micro_char --config configs/micro_char.yaml --max-steps 200 --run-name phase01a_overfit_smoke` |
| `data/tokenized/phase02a_smoke/` | Tokenized train/validation arrays and sidecars. | `uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2` |
| `data/tokenized/tokenizers/phase02a-byte-bpe-bilingual.json` | Generated tokenizer model JSON. | `uv run python -m tokenizer.train_report --config configs/tokenizer_bilingual.yaml --output docs/tokenizer_report.md` |
| `experiments/runs/phase03a_transformer_micro/` | Micro Transformer metrics, checkpoint, samples, manifest, and the default inference-smoke checkpoint. | `uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20` or bootstrap with `uv run python -m inference.generate --config configs/inference_smoke.yaml --prompt hello --max-new-tokens 16 --seed 123` |
| `experiments/runs/phase04a_tiny_smoke/` | Tiny pretraining checkpoints, metrics, samples, `eval_report.md`, and manifest. | `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke` then `uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --checkpoint experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt --output docs/phase04a_tiny_report.md` |
| `experiments/runs/phase05a_kgpt30m_smoke/` | 30M scale-gate checkpoints, metrics, fixed-prompt samples, checkpoint metadata, and manifest. | `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 --run-name phase05a_kgpt30m_smoke` then `uv run python -m eval.compare_runs --manifest docs/phase05a_scaling_manifest.json --output docs/phase05a_scaling_report.md` |
| `experiments/runs/phase06a_sft_smoke/` | SFT checkpoints, metrics, prompt-template metadata, `eval_report.md`, and manifest. | `uv run python -m train.sft --config configs/sft_smoke.yaml --max-steps 50 --run-name phase06a_sft_smoke` then `uv run python -m eval.sft_compare --config configs/sft_eval.yaml --output docs/phase06a_sft_eval.md` |
| `data/processed/corpus_v01_smoke/` | PHASE-10B deterministic smoke processed corpus with JSONL records and manifest. | `uv run python -m corpus.clean --config configs/corpus_v01.yaml --smoke --output data/processed/corpus_v01_smoke` |
| `data/tokenized/corpus_v01_tokenizers/byte_bpe_4k.json` | PHASE-10D selected corpus_v01 byte-BPE tokenizer model. | `uv run python -m tokenizer.train_report --config configs/tokenizer_corpus_v01.yaml --output docs/tokenizer_corpus_v01_report.md` |
| `data/tokenized/corpus_v01_smoke/` | PHASE-10D train/validation/test token arrays and metadata for corpus_v01 smoke data. | `uv run python -m train.sample_batches --config configs/corpus_v01_tokenized.yaml --max-batches 2` |
| `experiments/runs/phase11a_kgpt30m_corpus_v01_smoke/` | PHASE-11A real-corpus 30M smoke checkpoints, metrics, samples, checkpoint metadata, and manifest. | `uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --max-steps 1000 --run-name phase11a_kgpt30m_corpus_v01_smoke` then `uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest_corpus_v01.json --output docs/phase11a_real_corpus_checkpoint_comparison.md` |

## Policy

- Commit only compact evidence summaries, configs, schemas, and validation outputs.
- Do not commit `*.pt`, `*.safetensors`, private corpora, processed corpora, tokenized arrays, or generated run trees.
- A report may reference ignored artifacts by path when the command index explains how to regenerate them.
- If a checkpoint is missing in a fresh clone, evaluators must label the row as summary-only rather than pretending to
  have run live checkpoint evaluation.
