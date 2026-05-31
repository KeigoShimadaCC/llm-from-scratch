import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { createGitHubCliAdapter, parseStatusCheckRollup } from '../src/adapters/github-cli-adapter.js';
import type { CommandExecutionOptions, CommandExecutionResult } from '../src/adapters/command-executor.js';

const makeCommandResult = (
  command: string,
  options: CommandExecutionOptions,
  status: CommandExecutionResult['status'],
): CommandExecutionResult => {
  const now = new Date('2026-01-01T00:00:00.000Z').toISOString();
  return {
    command: [command, ...(options.args ?? [])].join(' '),
    cwd: options.cwd,
    exitCode: status === 'pass' ? 0 : 1,
    startedAt: now,
    finishedAt: now,
    durationMs: 0,
    stdoutPath: options.stdoutPath,
    stderrPath: options.stderrPath,
    status,
  };
};

describe('GitHub CLI adapter', () => {
  it('parses statusCheckRollup states', () => {
    expect(parseStatusCheckRollup('{"statusCheckRollup":[]}')).toBe('none');
    expect(
      parseStatusCheckRollup(
        '{"statusCheckRollup":[{"status":"IN_PROGRESS","conclusion":""}]}',
      ),
    ).toBe('pending');
    expect(
      parseStatusCheckRollup(
        '{"statusCheckRollup":[{"status":"COMPLETED","conclusion":"SUCCESS"}]}',
      ),
    ).toBe('pass');
    expect(
      parseStatusCheckRollup(
        '{"statusCheckRollup":[{"status":"COMPLETED","conclusion":"FAILURE"}]}',
      ),
    ).toBe('fail');
  });

  it('recovers when gh pr checks runs before GitHub registers checks', async () => {
    const repoRoot = await mkdtemp(path.join(os.tmpdir(), 'agentic-gh-adapter-test-'));
    try {
      const adapter = createGitHubCliAdapter({
        executor: {
          async run(command, options) {
            const args = options.args ?? [];
            await mkdir(path.dirname(options.stdoutPath), { recursive: true });
            await mkdir(path.dirname(options.stderrPath), { recursive: true });
            if (args.includes('checks')) {
              await writeFile(options.stdoutPath, 'no checks reported on the branch');
              await writeFile(options.stderrPath, '');
              return makeCommandResult(command, options, 'pass');
            }
            if (args.includes('view')) {
              await writeFile(
                options.stdoutPath,
                '{"statusCheckRollup":[{"status":"COMPLETED","conclusion":"SUCCESS"}]}',
              );
              await writeFile(options.stderrPath, '');
              return makeCommandResult(command, options, 'pass');
            }
            await writeFile(options.stdoutPath, '');
            await writeFile(options.stderrPath, '');
            return makeCommandResult(command, options, 'fail');
          },
        },
      });

      const evidenceDir = path.join(repoRoot, 'evidence');
      const result = await adapter.watchChecks({
        repoRoot,
        prNumber: 2,
        evidenceDir,
        timeoutMs: 1000,
      });
      const persisted = await readFile(path.join(evidenceDir, 'checks.json'), 'utf8');

      expect(result.status).toBe('pass');
      expect(persisted).toContain('"status": "pass"');
    } finally {
      await rm(repoRoot, { recursive: true, force: true });
    }
  });
});
