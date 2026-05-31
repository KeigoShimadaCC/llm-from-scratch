# PHASE-07A fixed checkpoint evaluation

## Summary

- Manifest: `docs/checkpoint_manifest.json`
- Checkpoints compared: 4
- Live evaluated checkpoints: 1
- Summary-only checkpoints: 3
- Summary-only status means the ignored local checkpoint was absent.
- Summary-only rows use committed phase summary metrics and do not claim fresh generation samples.

## Metric Definitions

- `validation_loss`: Mean next-token loss for pretraining checkpoints or response-only loss for SFT checkpoints.
- `perplexity`: exp(validation_loss), using the same cap as training reports.
- `tokens_per_sec`: Last recorded throughput from the run metrics file when that ignored artifact is available.
- `memory_usage`: Last recorded memory payload from run metrics; CPU runs report device only.
- `repetition_rate`: Fraction of repeated 3-grams in generated completion text.
- `average_completion_tokens`: Mean number of newly generated tokens across fixed prompts and toy tasks.
- `average_entropy`: Mean next-token entropy at the prompt boundary.
- `exact_match_rate`: Fraction of toy instruction completions exactly matching the expected response.

## Fixed Prompt Set

- `english-continuation` (english): The model learns
- `japanese-continuation` (japanese): 小さなモデル
- `technical-explanation` (technical): A transformer model uses attention because
- `instruction-gradient` (instruction): Q:Explain gradient descent in simple terms.
A:
- `bilingual-translation` (bilingual): Translate this short sentence into Japanese: training loop
- `copy-probe` (copy): Copy exactly: kgpt

## Exact-Match Toy Tasks

- `toy-say-hi`: `Q:say hi
A:` -> `hi`
- `toy-repeat-cat`: `Q:repeat cat
A:` -> `cat`
- `toy-answer-blue`: `Q:answer blue
A:` -> `blue`
- `toy-two-plus-two`: `Q:what is 2 plus 2?
A:` -> `4`

## Checkpoint Reports

### PHASE-03A micro Transformer

- Phase: PHASE-03A
- Kind: transformer_smoke
- Status: live_evaluated
- Status reason: Checkpoint artifact was present and loaded.
- Checkpoint: `experiments/runs/phase03a_transformer_micro/checkpoint_last.pt`
- Parameters: 29,728
- Validation loss: 1.0930
- Perplexity: 2.9831
- Tokens/sec: 14762.5734
- Repetition rate: 0.7322
- Average entropy: 1.4689
- Toy exact match: 0.00%
- Failure classes: bad_token_boundaries, instruction_ignored, pure_gibberish, repetition_loop, syntax_without_semantics
- Reproduce command: `uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20`

Leakage and memorization:
- `docs/phase02a_data_manifest.json`: overlap_count=0 (exact normalized text sha256 intersection between train and validation splits)
- Memorized generated fragments: 0

Samples:
- `english-continuation` english: completion=`e larger trainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrain`; repetition=0.833; entropy=1.208; top=e:0.627, ト:0.142, s:0.140
- `japanese-continuation` japanese: completion=`e����`; repetition=0.333; entropy=1.152; top=e:0.712, �:0.173, @:0.020
- `technical-explanation` technical: completion=`ee����������������������`; repetition=0.864; entropy=2.362; top=e:0.361, <blank>:0.114, rain:0.107
- `instruction-gradient` instruction: completion=`keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF-`; repetition=0.957; entropy=1.683; top=keeps every UTF-:0.352, �:0.335, �:0.110
- `bilingual-translation` bilingual: completion=``; repetition=0.000; entropy=1.789; top=<blank>:0.508, text:0.205, e:0.077
- `copy-probe` copy: completion=`keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every  keeps every`; repetition=0.957; entropy=0.827; top=keeps every:0.835, keeps every UTF-8 character recoverable.:0.052, �:0.032
- `toy-say-hi` toy_instruction: completion=`e larger trainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrainrain`; repetition=0.833; entropy=0.647; top=e:0.664, ::0.335, Byte fallback keeps every UTF-8 character recoverable.:0.000
- `toy-repeat-cat` toy_instruction: completion=`���ます。�す。�す。�す。����������������`; repetition=0.724; entropy=2.677; top=�:0.267, ��:0.160, を:0.090
- `toy-answer-blue` toy_instruction: completion=`:T����������������������`; repetition=0.864; entropy=0.141; top=::0.982, T:0.003, o:0.002
- `toy-two-plus-two` arithmetic_toy: completion=`keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF- keeps every UTF-`; repetition=0.957; entropy=2.204; top=keeps every UTF-:0.249, ::0.231, �:0.153


### PHASE-04A tiny pretraining

- Phase: PHASE-04A
- Kind: pretrain
- Status: summary_only_missing_ignored_checkpoint
- Status reason: Ignored checkpoint artifact was not present in this checkout.
- Checkpoint: `experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt`
- Parameters: 5,633,536
- Validation loss: 4.0593
- Perplexity: 57.9355
- Tokens/sec: n/a
- Repetition rate: n/a
- Average entropy: n/a
- Toy exact match: n/a
- Failure classes: repetition_risk, tiny_corpus_memorization_risk
- Reproduce command: `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke`

Leakage and memorization:
- `docs/phase04a_data_manifest.json`: overlap_count=0 (exact normalized text sha256 intersection between train and validation splits)
- Memorized generated fragments: n/a

Samples: not generated because the checkpoint artifact was absent.


### PHASE-05A kgpt-30m

- Phase: PHASE-05A
- Kind: pretrain
- Status: summary_only_missing_ignored_checkpoint
- Status reason: Ignored checkpoint artifact was not present in this checkout.
- Checkpoint: `experiments/runs/phase05a_kgpt30m_smoke/checkpoint_last.pt`
- Parameters: 31,734,272
- Validation loss: 11.4647
- Perplexity: 95294.2980
- Tokens/sec: 254.0450
- Repetition rate: n/a
- Average entropy: n/a
- Toy exact match: n/a
- Failure classes: undersampled_30m_run, tiny_corpus_memorization_risk
- Reproduce command: `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 --run-name phase05a_kgpt30m_smoke`

Leakage and memorization:
- `docs/phase04a_data_manifest.json`: overlap_count=0 (exact normalized text sha256 intersection between train and validation splits)
- `docs/phase05a_data_mixture_manifest.json`: overlap_count=not_reported (manifest does not expose a train/validation leakage_check field)
- Memorized generated fragments: n/a

Samples: not generated because the checkpoint artifact was absent.


### PHASE-06A instruction tuned

- Phase: PHASE-06A
- Kind: sft
- Status: summary_only_missing_ignored_checkpoint
- Status reason: Ignored checkpoint artifact was not present in this checkout.
- Checkpoint: `experiments/runs/phase06a_sft_smoke/checkpoint_last.pt`
- Parameters: 31,734,272
- Validation loss: 63.1710
- Perplexity: 485165195.4098
- Tokens/sec: n/a
- Repetition rate: n/a
- Average entropy: n/a
- Toy exact match: n/a
- Failure classes: heldout_validation_regression, narrow_command_memorization
- Reproduce command: `uv run python -m train.sft --config configs/sft_smoke.yaml --max-steps 50 --run-name phase06a_sft_smoke`

Leakage and memorization:
- `docs/phase06a_instruction_data_manifest.json`: overlap_count=not_reported (manifest does not expose a train/validation leakage_check field)
- Memorized generated fragments: n/a

Samples: not generated because the checkpoint artifact was absent.

