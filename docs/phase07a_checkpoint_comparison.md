# PHASE-07A Checkpoint Comparison

## Scope

- Eval config: `configs/eval_fixed_prompts.yaml`
- Checkpoint manifest schema: 1
- Policy: Ignored checkpoint artifacts are evaluated live when present. Fresh clones and CI use committed phase summaries and mark those rows summary-only.

## Comparison Table

| Checkpoint | Phase | Status | Parameters | Validation loss | Perplexity | Tokens/sec | Toy exact match | Failure classes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PHASE-03A micro Transformer | PHASE-03A | live_evaluated | 29,728 | 1.0930 | 2.9831 | 18898.2539 | 0.00% | bad_token_boundaries, instruction_ignored, pure_gibberish, repetition_loop, syntax_without_semantics |
| PHASE-04A tiny pretraining | PHASE-04A | live_evaluated | 5,633,536 | 4.8643 | 129.5822 | 2103.6298 | 0.00% | bad_token_boundaries, instruction_ignored, language_mixing, repetition_loop, syntax_without_semantics |
| PHASE-05A kgpt-30m | PHASE-05A | live_evaluated | 31,734,272 | 13.8912 | 1078680.0794 | 245.1161 | 0.00% | bad_token_boundaries, instruction_ignored, repetition_loop |
| PHASE-06A instruction tuned | PHASE-06A | live_evaluated | 31,734,272 | 11.2783 | 79083.2302 | 1971.5915 | 50.00% | instruction_ignored, mode_collapse, pure_gibberish |

## Failure Analysis

- PHASE-03A micro Transformer: bad_token_boundaries, instruction_ignored, pure_gibberish, repetition_loop, syntax_without_semantics
- PHASE-04A tiny pretraining: bad_token_boundaries, instruction_ignored, language_mixing, repetition_loop, syntax_without_semantics
- PHASE-05A kgpt-30m: bad_token_boundaries, instruction_ignored, repetition_loop
- PHASE-06A instruction tuned: instruction_ignored, mode_collapse, pure_gibberish

## Missing Ignored Artifacts

- None; all listed checkpoints were live evaluated.

## Comparison Notes

- The corpus is repo-authored and tiny, so these reports compare wiring, scale behavior, and failure modes rather than general language quality.
- Micro, tiny, 30M+, and SFT checkpoints use one prompt and metric schema.
- SFT exact-match evidence is limited to the compact command fixture and does not override the held-out SFT validation regression.

## Phase Gate

The evaluator uses one schema for micro, tiny, 30M+, and SFT checkpoints. When ignored checkpoints are present, it
loads and samples them directly; when they are absent, the report is explicitly marked summary-only and limited to
committed phase metrics.
