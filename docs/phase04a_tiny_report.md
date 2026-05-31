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

- step 0: validation_loss=164.5951, perplexity=485165195.41
- step 20: validation_loss=10.9378, perplexity=56263.79
- step 40: validation_loss=7.4966, perplexity=1801.98
- step 60: validation_loss=6.0518, perplexity=424.88
- step 80: validation_loss=5.6432, perplexity=282.37
- step 100: validation_loss=5.4892, perplexity=242.07
- step 120: validation_loss=4.8376, perplexity=126.16
- step 140: validation_loss=4.4629, perplexity=86.74
- step 160: validation_loss=4.2334, perplexity=68.95
- step 180: validation_loss=4.1696, perplexity=64.69
- step 200: validation_loss=4.0593, perplexity=57.94

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

## Fixed Prompt Samples From Final Checkpoint

### Prompt: `hello`

```text
hellored los.
```
### Prompt: `The model learns`

```text
The model learns.
```
### Prompt: `training loop`

```text
training loopred lost lost los.
```
### Prompt: `小さなモデル`

```text
小さなモデル�������������������������������
```
### Prompt: `検証データ`

```text
検証データななイ���なンンンンル�����������
```

## Resume Behavior

`checkpoint_last.pt` stores model state, optimizer state, current step, initial validation loss, best validation
loss, best step, and the full pretraining config. `python -m train.pretrain --resume` reloads that state from the
configured run directory and continues to the configured target step.

## Failure Modes

- The corpus is intentionally tiny and repo-authored, so sample quality is evidence of wiring and memorization
  behavior, not general language ability.
- Validation examples share the corpus style but are exact-hash separated from training records; remaining risk is
  distribution similarity, not duplicate leakage.
- Generated text may repeat because PHASE-04A does not add repetition penalties, nucleus sampling, or instruction
  tuning.

## Next Scale Recommendation

Move to PHASE-05A only if validation loss improves by the predeclared threshold and checkpoint_best.pt exists.
