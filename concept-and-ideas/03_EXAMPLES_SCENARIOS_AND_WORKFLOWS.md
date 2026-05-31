# Examples, Scenarios, And Workflows

## Primary Workflow

1. Define a phase in `phase-plans/`.
2. Run the phase in manual dry-run mode to inspect prompts and scope.
3. Run supervised mode only after the phase plan and allowed paths are correct.
4. Validate deterministically.
5. Record results, blockers, and next work in `PROGRESS.md`.

## LLM Lab Workflow

1. Prove the project skeleton with dummy token data.
2. Overfit a tiny character dataset.
3. Train or configure a real tokenizer.
4. Implement the decoder-only Transformer.
5. Run tiny pretraining, then small pretraining.
6. Add instruction tuning only after base language behavior exists.
7. Evaluate every serious checkpoint with fixed probes and failure analysis.
8. Add Mac-native inference and optimization last.

## Phase Mapping

- `PHASE-00B`: repo and lab foundation.
- `PHASE-01A`: MicroGPT character LM.
- `PHASE-02A`: tokenizer and dataset pipeline.
- `PHASE-03A`: core decoder-only Transformer.
- `PHASE-04A`: tiny pretraining.
- `PHASE-05A`: small practical model.
- `PHASE-06A`: instruction tuning.
- `PHASE-07A`: evaluation and failure analysis.
- `PHASE-08A`: Mac-native inference.
- `PHASE-09A`: final write-up.

## Agentic Workflow

- Planner proposes scoped tasks from the active phase plan.
- Executor edits only the allowed paths.
- Rechecker compares the diff against acceptance criteria and validation.
- Human/orchestrator owns architecture, data choices, and interpretation of training results.
