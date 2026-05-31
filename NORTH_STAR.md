# NORTH_STAR.md

The canonical North Star for this project is:

`docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md`

That document defines the project identity, from-scratch boundary, technical specification, model ladder, phase roadmap, data/evaluation principles, and final definition of done. Treat it as the source of truth before editing architecture, training, tokenizer, evaluation, or inference work.

Current final positioning is tracked in `docs/FINAL_WRITEUP.md`: the project is partially achieved with documented
fallback. The scratch-owned PyTorch path, tokenizer/data pipeline, training runs, evaluation reports, instruction
tuning smoke path, and local inference path exist. MLX inference and PyTorch-vs-MLX parity remain deferred, and the
50M/100M stretch models are dry-run targets rather than trained checkpoints.
