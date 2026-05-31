import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { ShellAgentAdapter, renderAgentCommandTemplate } from '../src/adapters/agent-adapters.js';
import type { CommandExecutionResult, CommandExecutor } from '../src/adapters/command-executor.js';

const passingResult = (command: string, cwd: string, stdoutPath: string, stderrPath: string): CommandExecutionResult => ({
  command,
  cwd,
  exitCode: 0,
  startedAt: '2026-06-01T00:00:00.000Z',
  finishedAt: '2026-06-01T00:00:01.000Z',
  durationMs: 1000,
  stdoutPath,
  stderrPath,
  status: 'pass',
});

describe('agent adapters', () => {
  it('renders evidence and output path variables', () => {
    const command = renderAgentCommandTemplate(
      "codex exec --add-dir '{{EVIDENCE_DIR}}' --output-last-message '{{OUTPUT_PATH}}' '{{PHASE_ID}}'",
      {
        WORKSPACE: '/repo-wt',
        PROMPT_PATH: '/repo/runs/prompt.md',
        OUTPUT_PATH: '/repo/runs/output.log',
        EVIDENCE_DIR: '/repo/runs/phase',
        PHASE_ID: 'PHASE-01A',
      },
    );

    expect(command).toContain("--add-dir '/repo/runs/phase'");
    expect(command).toContain("--output-last-message '/repo/runs/output.log'");
    expect(command).toContain("'PHASE-01A'");
  });

  it('prefers structured output written to outputPath over streamed stdout', async () => {
    const workspace = await mkdtemp(path.join(os.tmpdir(), 'agent-adapter-output-file-'));
    try {
      const evidenceDir = path.join(workspace, 'evidence');
      const outputPath = path.join(evidenceDir, 'executor-output.log');
      const reportText = JSON.stringify({
        schemaVersion: 1,
        phase: 'PHASE-01A',
        status: 'pass',
        filesChanged: ['kgpt/example.py'],
      });
      const executor: CommandExecutor = {
        async run(command, options) {
          await mkdir(path.dirname(options.stdoutPath), { recursive: true });
          await mkdir(path.dirname(options.stderrPath), { recursive: true });
          await writeFile(options.stdoutPath, 'streaming progress only');
          await writeFile(options.stderrPath, '');
          await writeFile(outputPath, reportText);
          return passingResult(command, options.cwd, options.stdoutPath, options.stderrPath);
        },
      };
      const adapter = new ShellAgentAdapter(
        {
          provider: 'shell',
          commandTemplate: 'fake-agent --output {{OUTPUT_PATH}}',
        },
        executor,
      );

      const result = await adapter.run({
        role: 'executor',
        workspace,
        promptPath: path.join(workspace, 'prompt.md'),
        outputPath,
        evidenceDir,
        phaseId: 'PHASE-01A',
      });

      expect(result.status).toBe('pass');
      await expect(readFile(outputPath, 'utf8')).resolves.toBe(reportText);
      await expect(readFile(path.join(evidenceDir, 'agent-results', 'executor-report.json'), 'utf8')).resolves.toContain(
        'kgpt/example.py',
      );
    } finally {
      await rm(workspace, { recursive: true, force: true });
    }
  });

  it('falls back to stdout when the agent does not write outputPath', async () => {
    const workspace = await mkdtemp(path.join(os.tmpdir(), 'agent-adapter-stdout-'));
    try {
      const evidenceDir = path.join(workspace, 'evidence');
      const outputPath = path.join(evidenceDir, 'planner-output.log');
      const reportText = JSON.stringify({
        schemaVersion: 1,
        phase: 'PHASE-01A',
        status: 'pass',
        planAcceptanceRecommendation: 'accept',
      });
      const executor: CommandExecutor = {
        async run(command, options) {
          await mkdir(path.dirname(options.stdoutPath), { recursive: true });
          await mkdir(path.dirname(options.stderrPath), { recursive: true });
          await writeFile(options.stdoutPath, reportText);
          await writeFile(options.stderrPath, '');
          return passingResult(command, options.cwd, options.stdoutPath, options.stderrPath);
        },
      };
      const adapter = new ShellAgentAdapter(
        {
          provider: 'shell',
          commandTemplate: 'fake-agent',
        },
        executor,
      );

      const result = await adapter.run({
        role: 'planner',
        workspace,
        promptPath: path.join(workspace, 'prompt.md'),
        outputPath,
        evidenceDir,
        phaseId: 'PHASE-01A',
      });

      expect(result.status).toBe('pass');
      await expect(readFile(outputPath, 'utf8')).resolves.toBe(reportText);
    } finally {
      await rm(workspace, { recursive: true, force: true });
    }
  });
});
