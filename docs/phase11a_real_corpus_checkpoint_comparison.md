# PHASE-11A Real-Corpus 30M Checkpoint Comparison

## Scope

- Eval config: `configs/eval_fixed_prompts.yaml`
- Checkpoint manifest schema: 1
- Policy: Ignored corpus_v01 checkpoint artifacts are evaluated live when present. Fresh clones and CI use this committed summary and mark the row summary-only.

## Comparison Table

| Checkpoint | Phase | Status | Parameters | Validation loss | Perplexity | Tokens/sec | Toy exact match | Failure classes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PHASE-11A kgpt-30m corpus_v01 smoke | PHASE-11A | live_evaluated | 31,692,800 | 30.4845 | 485165195.4098 | 344.5282 | 0.00% | instruction_ignored, mode_collapse, pure_gibberish |

## Live Sample Snapshots

### PHASE-11A kgpt-30m corpus_v01 smoke

Samples:
- `english-continuation` english: completion=`.`; repetition=0.000; entropy=0.008; top=.:0.999, <blank>:0.001, <blank>:0.000
- `japanese-continuation` japanese: completion=`めの短い文章です。`; repetition=0.000; entropy=0.030; top=�:0.996, �:0.004, �:0.000
- `technical-explanation` technical: completion=`y are trained on tokens.`; repetition=0.000; entropy=0.000; top=y:1.000, l:0.000, �:0.000
- `instruction-gradient` instruction: completion=``; repetition=0.000; entropy=0.840; top=<blank>:0.760, ra:0.173, t:0.027
- `bilingual-translation` bilingual: completion=``; repetition=0.000; entropy=0.748; top=<blank>:0.811, �:0.097, s:0.046
- `copy-probe` copy: completion=`trained on tokens.`; repetition=0.000; entropy=0.352; top=t:0.922, t:0.041, .:0.028
- `toy-say-hi` toy_instruction: completion=`めの短い文章です。`; repetition=0.000; entropy=0.155; top=�:0.965, �:0.035, 注:0.000
- `toy-repeat-cat` toy_instruction: completion=`めの短い文章です。`; repetition=0.000; entropy=1.410; top=�:0.624, �:0.157, �:0.075
- `toy-answer-blue` toy_instruction: completion=``; repetition=0.000; entropy=0.610; top=<blank>:0.873, �:0.047, �:0.039
- `toy-two-plus-two` arithmetic_toy: completion=``; repetition=0.000; entropy=1.340; top=<blank>:0.624, �:0.170, ra:0.098

## Failure Analysis

- PHASE-11A kgpt-30m corpus_v01 smoke: instruction_ignored, mode_collapse, pure_gibberish

## Missing Ignored Artifacts

- None; all listed checkpoints were live evaluated.

## Comparison Notes

- PHASE-11A is a smoke-sized real-corpus path over corpus_v01 local artifacts, not a full quality claim.
- The selected tokenizer is kgpt-corpus-v01-byte-bpe-4k from PHASE-10D; the smoke corpus caps the actual vocabulary at 312 tokens.
- English and Japanese fixed prompts are educational probes used to observe training dynamics and failure modes.

## Phase Gate

The evaluator uses one schema for micro, tiny, 30M+, and SFT checkpoints. When ignored checkpoints are present, it
loads and samples them directly; when they are absent, the report is explicitly marked summary-only and limited to
committed phase metrics.
