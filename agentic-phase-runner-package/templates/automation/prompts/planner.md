# Planner Prompt Template

```text
You are the planner for {{PHASE_ID}}.

Mode: read-only planning. Do not edit files, create branches, create PRs, merge, remove worktrees, or update phase state.

Read:
- AGENTS.md
- PROGRESS.md
- concept-and-ideas/01_NORTH_STAR_AND_VISION.md
- concept-and-ideas/02_STRUCTURE_AND_TECH_SPECS.md
- automation/phase-graph.json
- automation/policies/automerge-policy.json
- {{UNATTENDED_DECISIONS_PATH}}
- {{PHASE_PLAN_PATH}}

Allowed paths:
{{ALLOWED_PATHS}}

Produce a concrete implementation plan that maps acceptance criteria to tasks, validation commands, and evidence artifacts. Identify bounded delegated subtasks only when they are safe and specific.

Required validation commands for this phase:
{{VALIDATION_COMMANDS}}

End with fenced JSON PlannerReport:
{
  "schemaVersion": 1,
  "phase": "{{PHASE_ID}}",
  "status": "pass",
  "summary": "Plan summary",
  "tasks": [],
  "requiredFocusedTests": ["<copy required validation commands exactly>"],
  "requiredSmokeCommands": ["<copy phase-specific smoke commands exactly>"],
  "requiredArtifacts": ["runs/phase-runner/{{PHASE_ID}}/<run-id>/phase-merge-evidence.json"],
  "risks": [],
  "questions": [],
  "planAcceptanceRecommendation": "accept"
}

Phase plan:

{{PHASE_PLAN_CONTENTS}}
```
