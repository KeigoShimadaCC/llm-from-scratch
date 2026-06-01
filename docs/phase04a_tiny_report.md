# PHASE-04A tiny pretraining fixed-prompt report

## Summary

- Checkpoint: `experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt`
- Model: `kgpt-tiny-5m`
- Parameter count: 5,633,536
- Initial validation loss: 164.5951
- Final validation loss: 4.0593
- Best validation loss: 4.0593 at step 200
- Loss improvement: 97.53%
- Predeclared threshold: 10.00%
- Gate: pass

## Dataset Source And License

- Source: phase04a_repo_authored_tiny_pretrain_v1
- License: Repo-authored fixture for this project; no third-party text source.
- Language mix: English, Japanese, and mixed Japanese/English training notes.
- Split method: Deterministic sha256 ordering by seed, record id, and normalized text hash with validation_fraction=0.20.
- Dedup strategy: Unicode NFKC normalization, line-ending normalization, trailing-space trim, then exact normalized text sha256 deduplication.
- Leakage check: 0 overlapping normalized text hashes.

## Tokenizer

- Tokenizer id: phase02a-byte-bpe-bilingual
- Algorithm: byte_bpe
- Vocabulary size: 377
- Byte fallback: True

## Model Config

- Context length: 32
- Embedding dimension: 256
- Layers: 7
- Heads: 8
- MLP hidden dimension: 1024
- Tied embeddings: True

## Run Budget And Hyperparameters

- Target steps: 200
- Target tokens: 12800
- Target wall clock: 30 minutes
- Hardware/device: Mac local CPU/MPS, default validation uses CPU for reproducibility. / `cpu`
- Optimizer: adamw, lr=0.001
- Scheduler: warmup=10, min_lr_factor=0.1
- Batch size: 2
- Gradient accumulation steps: 1
- Max grad norm: 1.0

## Validation Curve

- step 0: validation_loss=164.5951, perplexity=485165195.4098
- step 20: validation_loss=10.9378, perplexity=56263.7856
- step 40: validation_loss=7.4966, perplexity=1801.9756
- step 60: validation_loss=6.0518, perplexity=424.8798
- step 80: validation_loss=5.6432, perplexity=282.3713
- step 100: validation_loss=5.4892, perplexity=242.0669
- step 120: validation_loss=4.8376, perplexity=126.1645
- step 140: validation_loss=4.4629, perplexity=86.7399
- step 160: validation_loss=4.2334, perplexity=68.9481
- step 180: validation_loss=4.1696, perplexity=64.6908
- step 200: validation_loss=4.0593, perplexity=57.9355

## Sample Progression

### Prompt: `hello`

- step 0: `hellooooooooooooooooooooooooooooooooo`
- step 200: `hellored los.`

### Prompt: `The model learns`

- step 0: `The model learnsssssssssssssssssssssssssssssssss`
- step 200: `The model learns.`

### Prompt: `training loop`

- step 0: `training looppppppppppppppppppppppppppppppppp`
- step 200: `training loopred lost lost los.`

### Prompt: `小さなモデル`

- step 0: `小さなモデル��������������������������������`
- step 200: `小さなモデル�������������������������������`

### Prompt: `検証データ`

- step 0: `検証データ��������������������������������`
- step 200: `検証データななイ���なンンンンル�����������`

## Fixed Prompt Eval Addendum

- Eval status: live_evaluated
- Eval validation loss: 4.8643
- Eval perplexity: 129.5822
- Toy exact match: 0.00%
- Failure classes: bad_token_boundaries, instruction_ignored, language_mixing, repetition_loop, syntax_without_semantics
- Run-local eval report: `experiments/runs/phase04a_tiny_smoke/eval_report.md`

## Resume Behavior

- `checkpoint_last.pt` and `checkpoint_best.pt` are both produced.
- The pretraining path supports resume through `checkpoint_last.pt`; the phase smoke run was generated from config.

## Failure Modes

- The fixture corpus is intentionally tiny and can encourage memorization.
- Japanese generation still shows byte-level token boundary artifacts.
- The model is useful as training-loop evidence, not as a general language model.

## Next Scale Recommendation

Proceed to PHASE-05A only as a scale-gate experiment with the same artifact policy and explicit quality limitations.
