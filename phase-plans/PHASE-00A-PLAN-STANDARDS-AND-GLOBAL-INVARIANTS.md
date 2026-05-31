# PHASE-00A - Plan Standards And Global Invariants

## Purpose

Use this file as the permanent checklist for all phase plans. This file is not an implementation phase in the graph.

## Required Sections

- Goal
- Scope
- Allowed Paths
- Forbidden Paths
- Tasks
- Acceptance Criteria
- Required Validation
- Risks
- Out Of Scope

## Global Invariants

- Read `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md` before changing project direction.
- Keep changes scoped to the active phase.
- Preserve secrets, private data, local checkpoints, and generated run artifacts.
- Update `PROGRESS.md`.
- Run validation before marking done.
- Record blockers honestly.
- Do not use pretrained model wrappers in the scratch-core training path.
- Prove core training behavior on tiny data before scaling.
