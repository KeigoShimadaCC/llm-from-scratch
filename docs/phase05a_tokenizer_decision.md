# PHASE-05A Tokenizer Decision

## Decision

Use `phase02a-byte-bpe-bilingual` as the provisional final tokenizer for PHASE-05A 30M+ training.

## Rationale

- It is repo-owned and deterministic, with no external model dependency.
- It preserves every UTF-8 byte through byte fallback, so English, Japanese, and mixed-script text roundtrip.
- PHASE-02A documented source provenance, split/dedup behavior, token file format, and leakage checks.
- PHASE-04A successfully trained a 5.63M-parameter model with this tokenizer and produced a reproducible report.

## Known Limits

- The tokenizer was trained from a tiny repo-authored corpus, so segmentation quality is not representative of a production bilingual tokenizer.
- Japanese text fragments heavily under tiny-data byte BPE, which affects effective context length.
- This remains a provisional educational tokenizer until larger PHASE-05A or PHASE-07A evidence justifies training a larger tokenizer.

## Approval

Under the unattended decision policy, this tokenizer is approved for PHASE-05A because it is the only fully repo-owned tokenizer with completed provenance and successful tiny-pretraining evidence.
