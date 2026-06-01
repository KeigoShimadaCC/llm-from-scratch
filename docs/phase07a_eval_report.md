# PHASE-07A fixed checkpoint evaluation

## Summary

- Manifest: `docs/checkpoint_manifest.json`
- Checkpoints compared: 4
- Live evaluated checkpoints: 4
- Summary-only checkpoints: 0
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
- Tokens/sec: 18898.2539
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
- Status: live_evaluated
- Status reason: Checkpoint artifact was present and loaded.
- Checkpoint: `experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt`
- Parameters: 5,633,536
- Validation loss: 4.8643
- Perplexity: 129.5822
- Tokens/sec: 2103.6298
- Repetition rate: 0.1794
- Average entropy: 4.0619
- Toy exact match: 0.00%
- Failure classes: bad_token_boundaries, instruction_ignored, language_mixing, repetition_loop, syntax_without_semantics
- Reproduce command: `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke`

Leakage and memorization:
- `docs/phase04a_data_manifest.json`: overlap_count=0 (exact normalized text sha256 intersection between train and validation splits)
- Memorized generated fragments: 0

Samples:
- `english-continuation` english: completion=`.`; repetition=0.000; entropy=3.926; top=.:0.079, t:0.067, c:0.067
- `japanese-continuation` japanese: completion=`�����������������������`; repetition=0.952; entropy=3.520; top=�:0.200, �:0.100, �:0.076
- `technical-explanation` technical: completion=`d lored lored lored lore`; repetition=0.000; entropy=2.789; top=d:0.294, ,:0.238, s:0.097
- `instruction-gradient` instruction: completion=`ing lored lored lored lor`; repetition=0.000; entropy=4.511; top=in:0.038, e:0.038, i:0.038
- `bilingual-translation` bilingual: completion=`red lored lored lored lo`; repetition=0.000; entropy=3.120; top=r:0.355, t:0.067, <blank>:0.057
- `copy-probe` copy: completion=`sa s sad los.`; repetition=0.000; entropy=4.087; top=s:0.091, <blank>:0.080, �:0.077
- `toy-say-hi` toy_instruction: completion=`なル�������������������`; repetition=0.842; entropy=4.635; top=�:0.062, s:0.041, �:0.032
- `toy-repeat-cat` toy_instruction: completion=`he mpred los.`; repetition=0.000; entropy=4.711; top=h:0.052, �:0.031, �:0.027
- `toy-answer-blue` toy_instruction: completion=`� los.`; repetition=0.000; entropy=4.598; top=�:0.063, �:0.044, �:0.038
- `toy-two-plus-two` arithmetic_toy: completion=`宮 los.`; repetition=0.000; entropy=4.721; top=�:0.037, �:0.031, �:0.031


### PHASE-05A kgpt-30m

- Phase: PHASE-05A
- Kind: pretrain
- Status: live_evaluated
- Status reason: Checkpoint artifact was present and loaded.
- Checkpoint: `experiments/runs/phase05a_kgpt30m_smoke/checkpoint_last.pt`
- Parameters: 31,734,272
- Validation loss: 13.8912
- Perplexity: 1078680.0794
- Tokens/sec: 245.1161
- Repetition rate: 0.9547
- Average entropy: 3.1918
- Toy exact match: 0.00%
- Failure classes: bad_token_boundaries, instruction_ignored, repetition_loop
- Reproduce command: `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 --run-name phase05a_kgpt30m_smoke`

Leakage and memorization:
- `docs/phase04a_data_manifest.json`: overlap_count=0 (exact normalized text sha256 intersection between train and validation splits)
- `docs/phase05a_data_mixture_manifest.json`: overlap_count=not_reported (manifest does not expose a train/validation leakage_check field)
- Memorized generated fragments: 0

Samples:
- `english-continuation` english: completion=`ssssssssssssssssssssssss`; repetition=0.955; entropy=0.800; top=s:0.893, �:0.005, r:0.005
- `japanese-continuation` japanese: completion=`������������������������`; repetition=0.955; entropy=4.077; top=�:0.087, #:0.077, �:0.064
- `technical-explanation` technical: completion=`eeeeeeeeeeeeeeeeeeeeeeee`; repetition=0.955; entropy=2.753; top=e:0.482, d:0.067, �:0.042
- `instruction-gradient` instruction: completion=`########################`; repetition=0.955; entropy=4.252; top=#:0.070, �:0.053, c:0.044
- `bilingual-translation` bilingual: completion=`pppppppppppppppppppppppp`; repetition=0.955; entropy=1.714; top=p:0.728, �:0.024, r:0.015
- `copy-probe` copy: completion=`tttttttttttttttttttttttt`; repetition=0.955; entropy=1.303; top=t:0.805, �:0.013, a:0.010
- `toy-say-hi` toy_instruction: completion=`alalalalalalalalalalalalalalalalalalalalalalalal`; repetition=0.957; entropy=4.288; top=al:0.040, m:0.036, s:0.034
- `toy-repeat-cat` toy_instruction: completion=`########################`; repetition=0.955; entropy=4.242; top=#:0.055, �:0.048, r:0.048
- `toy-answer-blue` toy_instruction: completion=`mmmmmmmmmmmmmmmmmmmmmmmm`; repetition=0.955; entropy=4.208; top=m:0.065, �:0.061, #:0.040
- `toy-two-plus-two` arithmetic_toy: completion=`rrrrrrrrrrrrrrrrrrrrrrrr`; repetition=0.955; entropy=4.280; top=r:0.045, e:0.038, #:0.038


### PHASE-06A instruction tuned

- Phase: PHASE-06A
- Kind: sft
- Status: live_evaluated
- Status reason: Checkpoint artifact was present and loaded.
- Checkpoint: `experiments/runs/phase06a_sft_smoke/checkpoint_last.pt`
- Parameters: 31,734,272
- Validation loss: 11.2783
- Perplexity: 79083.2302
- Tokens/sec: 1971.5915
- Repetition rate: 0.0000
- Average entropy: 0.0358
- Toy exact match: 50.00%
- Failure classes: instruction_ignored, mode_collapse, pure_gibberish
- Reproduce command: `uv run python -m train.sft --config configs/sft_smoke.yaml --max-steps 50 --run-name phase06a_sft_smoke`

Leakage and memorization:
- `docs/phase06a_instruction_data_manifest.json`: overlap_count=not_reported (manifest does not expose a train/validation leakage_check field)
- Memorized generated fragments: 0

Samples:
- `english-continuation` english: completion=``; repetition=0.000; entropy=0.137; top=<blank>:0.974, s:0.019, e:0.007
- `japanese-continuation` japanese: completion=`cat`; repetition=0.000; entropy=0.072; top=c:0.986, i:0.013, k:0.000
- `technical-explanation` technical: completion=``; repetition=0.000; entropy=0.037; top=<blank>:0.994, e:0.005, s:0.001
- `instruction-gradient` instruction: completion=`cat`; repetition=0.000; entropy=0.000; top=c:1.000, e:0.000, s:0.000
- `bilingual-translation` bilingual: completion=`e`; repetition=0.000; entropy=0.079; top=e:0.985, <blank>:0.015, c:0.000
- `copy-probe` copy: completion=`e`; repetition=0.000; entropy=0.000; top=e:1.000, <blank>:0.000, s:0.000
- `toy-say-hi` toy_instruction: completion=`hi`; repetition=0.000; entropy=0.025; top=h:0.996, c:0.004, e:0.000
- `toy-repeat-cat` toy_instruction: completion=`cat`; repetition=0.000; entropy=0.000; top=c:1.000, s:0.000, k:0.000
- `toy-answer-blue` toy_instruction: completion=`s`; repetition=0.000; entropy=0.006; top=s:0.999, <blank>:0.001, b:0.000
- `toy-two-plus-two` arithmetic_toy: completion=`cat`; repetition=0.000; entropy=0.002; top=c:1.000, s:0.000, b:0.000

