# PHASE-00A - Plan Standards And Global Invariants

## Purpose

Use this file as the permanent checklist for all phase plans. This file is not an implementation phase in the graph.

## Required Sections

- Goal
- Prerequisites
- Scope
- Allowed Paths
- Forbidden Paths
- Phase Dependencies
- Tasks
- Deliverables
- Evidence Artifacts
- Acceptance Criteria
- Required Validation
- Artifact Policy
- Human Decisions
- Phase Gate
- Risks
- Out Of Scope
- Deferred Backlog

## Global Invariants

- Read `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md` before changing project direction.
- Keep changes scoped to the active phase.
- Preserve secrets, private data, local checkpoints, and generated run artifacts.
- Update `PROGRESS.md`.
- Run validation before marking done.
- Record blockers honestly.
- Do not use pretrained model wrappers in the scratch-core training path.
- Prove core training behavior on tiny data before scaling.

## Artifact Policy

- Ignored local artifacts may be generated during a phase when they are required for validation or evidence.
- Ignored local artifacts must not be committed: `data/raw/**`, `data/processed/**`, `data/tokenized/**`, `experiments/runs/**`, `runs/**`, checkpoints, private corpora, and benchmark outputs.
- In phase plans, `Forbidden Paths` means forbidden to commit or manually edit as source changes; it does not prohibit required commands from generating ignored evidence artifacts when the phase's artifact policy allows them.
- Every serious run must produce an artifact manifest in docs or `PROGRESS.md` that names the ignored run directory and lists the evidence files created there.
- Any committed report that summarizes ignored artifacts must include the commands, config path, data manifest, checkpoint names, metrics, and limitations needed to reproduce or audit the claim.

## Data And Model Invariants

- Every dataset source must have provenance, license status, split strategy, dedup strategy, and contamination/leakage notes before it is used for training claims.
- Every tokenizer phase must report vocabulary size, English and Japanese token-per-sentence stats, unknown/byte fallback behavior, compression ratio, and bad segmentation examples.
- Every model scale-up must define its config, parameter count, run budget, hardware/device, validation metric, checkpoint policy, and fallback criteria before training.
- Evaluation assets should be introduced early and reused across phases rather than deferred to the final evaluation phase.

## Phase Dependency Policy

- Each executable phase must list predecessor phases, reusable assets it depends on, and downstream phases it unblocks.
- A dependency is satisfied only when its phase gate is complete or the human/orchestrator, or the committed unattended decision policy, accepts a documented blocker and fallback.
- Later phases must not weaken earlier evidence requirements to claim completion.

## Validation Policy

- Every phase starts with the global validation baseline: tests, lint, and `git diff --check`.
- When a phase creates a CLI, report generator, training smoke, evaluation flow, or benchmark, `Required Validation` must add the exact phase-specific command before the phase is marked complete.
- A phase cannot rely only on the global validation baseline when it adds behavior whose evidence comes from generated reports, run directories, evaluation outputs, or inference benchmarks.

## Human Decisions

The agent may implement scoped mechanics. In supervised mode the human/orchestrator owns these decisions; in unattended mode `automation/policies/unattended-decisions.json` pre-approves bounded choices and fallback rules:

- final dataset inclusion and licensing judgment;
- final tokenizer choice for serious training;
- scale-up go/no-go decisions;
- whether compute limits justify a documented fallback;
- interpretation of loss curves and generated samples;
- claims made in final write-ups.

## Phase Gate Policy

- A phase may be marked complete only after its required validation, evidence artifacts, artifact policy check, and human or unattended-policy decisions are resolved.
- Completion notes must name any ignored run artifacts used as evidence without committing those artifacts.

## Deferred Backlog Policy

- Deferred backlog items must be explicit, assigned to a later phase where possible, and excluded from current-phase completion claims.
