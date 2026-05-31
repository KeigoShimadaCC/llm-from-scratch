# PHASE-07A Checkpoint Comparison

## Scope

- Eval config: `configs/eval_fixed_prompts.yaml`
- Checkpoint manifest schema: 1
- Policy: Ignored checkpoint artifacts are evaluated live when present. Fresh clones and CI use committed phase summaries and mark those rows summary-only.

## Comparison Table

| Checkpoint | Phase | Status | Parameters | Validation loss | Perplexity | Tokens/sec | Toy exact match | Failure classes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PHASE-03A micro Transformer | PHASE-03A | live_evaluated | 29,728 | 1.0930 | 2.9831 | 14762.5734 | 0.00% | bad_token_boundaries, instruction_ignored, pure_gibberish, repetition_loop, syntax_without_semantics |
| PHASE-04A tiny pretraining | PHASE-04A | summary_only_missing_ignored_checkpoint | 5,633,536 | 4.0593 | 57.9355 | n/a | n/a | repetition_risk, tiny_corpus_memorization_risk |
| PHASE-05A kgpt-30m | PHASE-05A | summary_only_missing_ignored_checkpoint | 31,734,272 | 11.4647 | 95294.2980 | 254.0450 | n/a | undersampled_30m_run, tiny_corpus_memorization_risk |
| PHASE-06A instruction tuned | PHASE-06A | summary_only_missing_ignored_checkpoint | 31,734,272 | 63.1710 | 485165195.4098 | n/a | n/a | heldout_validation_regression, narrow_command_memorization |

## Failure Analysis

- PHASE-03A micro Transformer: bad_token_boundaries, instruction_ignored, pure_gibberish, repetition_loop, syntax_without_semantics
- PHASE-04A tiny pretraining: repetition_risk, tiny_corpus_memorization_risk
- PHASE-05A kgpt-30m: undersampled_30m_run, tiny_corpus_memorization_risk
- PHASE-06A instruction tuned: heldout_validation_regression, narrow_command_memorization

## Missing Ignored Artifacts

- `experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt`: Ignored checkpoint artifact was not present in this checkout.
- `experiments/runs/phase05a_kgpt30m_smoke/checkpoint_last.pt`: Ignored checkpoint artifact was not present in this checkout.
- `experiments/runs/phase06a_sft_smoke/checkpoint_last.pt`: Ignored checkpoint artifact was not present in this checkout.

## Comparison Notes

- The corpus is repo-authored and tiny, so these reports compare wiring, scale behavior, and failure modes rather than general language quality.
- Micro, tiny, 30M+, and SFT checkpoints use one prompt and metric schema.
- SFT exact-match evidence is limited to the compact command fixture and does not override the held-out SFT validation regression.

## Phase Gate

The evaluator uses one schema for micro, tiny, 30M+, and SFT checkpoints. When ignored checkpoints are present, it
loads and samples them directly; when they are absent, the report is explicitly marked summary-only and limited to
committed phase metrics.
