# Claim Evidence Audit

- Document: `docs/FINAL_WRITEUP.md`
- Claims audited: 11
- Unsupported claims: 0

| Claim | Declared status | Audit status | Missing evidence |
|---|---:|---:|---|
| The repo owns the core decoder-only Transformer implementation. | Supported | supported | none |
| The tokenizer and data pipeline are documented with provenance, split, dedup, byte fallback, and leakage evidence. | Supported | supported | none |
| Training runs are config-driven and produce ignored reproducible artifacts rather than committed checkpoints. | Supported | supported | none |
| A 5M-20M tiny model was trained from random initialization. | Supported | supported | none |
| A 30M+ model was trained locally from random initialization for the configured phase gate. | Supported | supported | none |
| The 50M and 100M stretch models were prepared as dry-run targets but not trained. | Partial / deferred | partial | none |
| A small instruction-tuned variant was produced, but evidence is limited to narrow probes and held-out regression is documented. | Partial | partial | none |
| Evaluation reports compare micro, tiny, 30M+, and SFT checkpoints under one schema. | Supported | supported | none |
| Local PyTorch CPU/MPS inference is implemented with sampling controls, KV-cache parity, and benchmark evidence. | Supported | supported | none |
| MLX inference and PyTorch-vs-MLX parity are not complete. | Deferred | partial | none |
| The final North Star status is partial rather than fully achieved. | Partial | partial | none |
