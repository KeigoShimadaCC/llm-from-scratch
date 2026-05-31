# Data Directory

This directory documents data conventions for KeigoGPT-Lab.

Local data is intentionally ignored by git:

- `data/raw/`
- `data/processed/`
- `data/tokenized/`

Every dataset used for training should have a documented source, license status, preprocessing command, train/validation split method, and tokenizer compatibility note. Keep clean, inspectable data ahead of raw scale.

Do not commit private corpora, downloaded datasets, tokenized binary files, or generated data artifacts.

PHASE-00B uses no real corpus. `train.dummy` creates fake token batches in memory for smoke testing only.

PHASE-01A uses a repo-authored synthetic character fixture embedded in `configs/micro_char.yaml`:

```text
hello microgpt.
```

The line is repeated to create a tiny, low-entropy overfit corpus. Source/provenance: authored for this
repository, with no external text source or third-party license dependency. The fixture is intentionally
small and is only for tokenizer roundtrip tests, training-loop smoke tests, deterministic generation checks,
and the first overfitting curve.
