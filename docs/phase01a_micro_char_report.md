# PHASE-01A MicroGPT Character LM Report

## Summary

PHASE-01A implements a minimal character-level language modeling loop using a repo-owned tokenizer,
a tiny context-conditioned PyTorch model, config-driven training, checkpointed run artifacts, and deterministic
generation.

The phase uses a context MLP baseline instead of a Transformer. This is deliberate: the phase gate is to prove
the training loop, checkpoint format, overfit behavior, and generation controls before moving to tokenizers and
full decoder-only Transformer blocks in later phases.

## Corpus

- Name: `micro_char_synthetic_hello_v1`
- Source: repo-authored synthetic fixture in `configs/micro_char.yaml`
- License/provenance: no external source; intended for this repository's test and smoke-training use
- Text pattern: repeated `hello microgpt.\n`
- Characters: 512
- Tokenizer: `char-v1-synthetic-hello`
- Vocabulary size: 14

This low-entropy corpus is intentionally tiny. It is not a language-quality dataset; it exists to make failure in
the training loop obvious and to provide a deterministic overfit target.

## Model And Config

- Model type: character context MLP next-character predictor
- Model name: `kgpt-char-context-mlp`
- Parameter count: 35,150
- Context length: 8 characters
- Embedding dim: 32
- Hidden dim: 128
- Dropout: 0.0
- Optimizer: AdamW
- Learning rate: 0.03
- Batch size: 64
- Seed: 123
- Device: CPU
- Training steps: 200
- Overfit threshold: train loss below 0.1

The model predicts the next character from a fixed-length left-padded context window. It is a baseline, not the
project's final architecture.

## Required Smoke Run

Command:

```bash
uv run python -m train.micro_char --config configs/micro_char.yaml --max-steps 200 --run-name phase01a_overfit_smoke
```

Ignored run evidence:

- `experiments/runs/phase01a_overfit_smoke/config.yaml`
- `experiments/runs/phase01a_overfit_smoke/metrics.jsonl`
- `experiments/runs/phase01a_overfit_smoke/samples.txt`
- `experiments/runs/phase01a_overfit_smoke/tokenizer.json`
- `experiments/runs/phase01a_overfit_smoke/checkpoint_last.pt`
- `experiments/runs/phase01a_overfit_smoke/manifest.json`

Results:

| Metric | Initial | Final |
| --- | ---: | ---: |
| Train loss | 2.714604377746582 | 0.00000252200834438554 |
| Validation loss | 2.723736524581909 | 0.0000016990157973850728 |
| Train perplexity | 15.098635533929771 | 1.0000025220115247 |

Overfit result: passed. Final train loss is below the 0.1 threshold.

## Generation

Required generation command:

```bash
uv run python -m inference.generate_char --checkpoint experiments/runs/phase01a_overfit_smoke/checkpoint_last.pt --prompt hello --seed 123 --max-new-tokens 32
```

Output text from the required command:

```text
hello microgpt.
hello microgpt.
hello
```

Training samples show the transition from untrained noise to the learned pattern:

```text
step=0 mode=greedy
hello hicicgeolr eiciectih emlglppmgc

step=200 mode=greedy_final
hello microgpt.
hello microgpt.
hello
```

Fixed prompts for later phases live in `eval/char_prompts.json`.

## Underfitting, Overfitting, And Temperature

Underfitting would look like persistently high train loss and samples that do not recover the repeated phrase.
The step 0 sample is an example of the pre-training behavior: it emits mostly arbitrary characters from the
vocabulary.

Overfitting is expected and desired in this phase. The final train loss is effectively zero because the corpus is
small, repeated, and low entropy. This proves that gradients, batching, checkpointing, and autoregressive
generation are wired correctly before the project moves to harder data.

Sampling temperature controls how sharply the model samples from its next-character probabilities. A temperature
of 0 or greedy decoding takes the highest-probability character every step. Higher temperatures keep lower
probability characters in play and can create variation. On this overfit corpus, even sampled generation at
temperature 0.8 usually follows the memorized phrase because the final distribution is extremely sharp.

## Known Failure Modes

- The baseline is not a Transformer and does not test causal self-attention.
- The corpus has almost no ambiguity, so validation loss is not a realistic generalization measure.
- Prompts can only contain characters present in the checkpoint vocabulary.
- Generation has no stop token and continues until `--max-new-tokens`.
- The fixture is too small to reveal tokenizer quality, long-context behavior, or natural language diversity.
