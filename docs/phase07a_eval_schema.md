# PHASE-07A Evaluation Schema

## Report Inputs

- Eval config: `configs/eval_fixed_prompts.yaml`
- Checkpoint manifest: `docs/checkpoint_manifest.json`
- Optional ignored checkpoint artifacts: `experiments/runs/**/checkpoint_last.pt`

The evaluator loads compatible checkpoints when the ignored artifacts exist locally. If a checkpoint is absent, reports
fall back to committed phase summaries and mark that row `summary_only_missing_ignored_checkpoint`.

## Metrics

- `validation_loss`: mean next-token loss for pretraining checkpoints or response-only loss for SFT checkpoints.
- `perplexity`: `exp(validation_loss)` with the same cap as training reports.
- `tokens_per_sec`: final recorded throughput from `metrics.jsonl` when available.
- `memory_usage`: final recorded memory payload from `metrics.jsonl` when available.
- `repetition_rate`: repeated 3-gram fraction in generated completion text.
- `average_completion_tokens`: mean generated-token count across fixed prompts and toy tasks.
- `average_entropy`: mean next-token entropy at the prompt boundary.
- `exact_match_rate`: fraction of toy instruction completions exactly matching the expected response.

## Failure Taxonomy

- `pure_gibberish`: generated completion is empty after normalization.
- `syntax_without_semantics`: completion is long enough to look sentence-like but lacks terminal punctuation.
- `repetition_loop`: repeated 3-gram rate is at least 25%.
- `mode_collapse`: multiple prompts produce too few unique completions.
- `language_mixing`: English and Japanese scripts mix outside bilingual prompts.
- `memorized_fragments`: generated completion contains a normalized source fragment.
- `bad_token_boundaries`: replacement characters appear in generated text.
- `instruction_ignored`: toy instruction completion does not exactly match the expected response.
- `false_factual_confidence`: generated text uses high-certainty language such as "definitely" or "always".

## Leakage And Memorization

Committed data manifests supply exact train/validation overlap counts. Live checkpoint evaluation additionally compares
generated completions against normalized repo-authored source records from the relevant tokenizer, pretraining, or SFT
config. Missing ignored checkpoints are not treated as evidence for live memorization behavior.

## Interpretation Policy

The comparison is an educational, Mac-local evidence artifact. It can support claims about reproducibility, scale gates,
and failure modes. It does not support claims about general model quality, benchmark performance, safety, or factuality.
