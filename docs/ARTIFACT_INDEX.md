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

## Ignored Local Artifact Locations

| Path | Contents | Recreate with |
|---|---|---|
| `experiments/runs/phase00b_smoke/` | Dummy foundation run artifacts. | `uv run python -m train.dummy --config configs/dummy.yaml --run-name phase00b_smoke` |
| `experiments/runs/phase01a_micro_char_overfit/` | Character LM metrics, checkpoint, samples, and manifest. | `uv run python -m train.micro_char --config configs/micro_char.yaml --run-name phase01a_micro_char_overfit` |
| `data/tokenized/phase02a_smoke/` | Tokenized train/validation arrays and sidecars. | `uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2` |
| `data/tokenized/tokenizers/phase02a-byte-bpe-bilingual.json` | Generated tokenizer model JSON. | `uv run python -m tokenizer.train_report --config configs/tokenizer_bilingual.yaml --output docs/tokenizer_report.md` |
| `experiments/runs/phase03a_transformer_micro/` | Micro Transformer metrics, checkpoint, samples, and manifest. | `uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20` |
| `experiments/runs/phase04a_tiny_smoke/` | Tiny pretraining checkpoints, metrics, samples, eval report, and manifest. | `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml` |
| `experiments/runs/phase05a_kgpt30m_smoke/` | 30M scale-gate checkpoints, metrics, samples, and manifest. | `uv run python -m train.pretrain --config configs/kgpt_30m.yaml` |
| `experiments/runs/phase06a_sft_smoke/` | SFT checkpoints, metrics, prompt-template metadata, and manifest. | `uv run python -m train.sft --config configs/sft_smoke.yaml` |
| `experiments/runs/phase08a_inference_smoke/` | Locally regenerated inference smoke checkpoint when missing. | `uv run python -m inference.generate --config configs/inference_smoke.yaml --prompt hello --max-new-tokens 16 --seed 123` |

## Policy

- Commit only compact evidence summaries, configs, schemas, and validation outputs.
- Do not commit `*.pt`, `*.safetensors`, private corpora, processed corpora, tokenized arrays, or generated run trees.
- A report may reference ignored artifacts by path when the command index explains how to regenerate them.
- If a checkpoint is missing in a fresh clone, evaluators must label the row as summary-only rather than pretending to
  have run live checkpoint evaluation.
