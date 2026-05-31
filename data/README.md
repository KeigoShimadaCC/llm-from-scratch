# Data Directory

This directory documents data conventions for KeigoGPT-Lab.

Local data is intentionally ignored by git:

- `data/raw/`
- `data/processed/`
- `data/tokenized/`

Every dataset used for training should have a documented source, license status, preprocessing command, train/validation split method, and tokenizer compatibility note. Keep clean, inspectable data ahead of raw scale.

Do not commit private corpora, downloaded datasets, tokenized binary files, or generated data artifacts.

PHASE-00B uses no real corpus. `train.dummy` creates fake token batches in memory for smoke testing only.
