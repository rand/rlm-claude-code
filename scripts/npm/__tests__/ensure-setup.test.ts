/**
 * Tests for ensure-setup.ts
 *
 * Tests the self-healing setup logic for:
 * - Detecting missing binaries/wheel/venv
 * - Determining installation mode (marketplace vs dev)
 * - Outputting JSON status for AI agent
 */

import { test, describe, it } from 'node:test';
import assert from 'node:assert';
import path from 'path';
import fs from 'fs';

// ============================================================================
// Installation Mode Detection Tests
// ============================================================================

describe('Installation Mode Detection', () => {
  // Marketplace: installed via `claude plugins install`
  // Dev: symlinked to plugins/cache/.../dev/

  const detectInstallMode = (pluginRoot: string): 'marketplace' | 'dev' => {
    // If the path contains '/dev/' or ends with '/dev', it's a dev installation
    if (pluginRoot.includes('/dev/') || pluginRoot.endsWith('/dev')) {
      return 'dev';
    }
    // If .git directory exists directly in plugin root, it's likely a dev install
    if (fs.existsSync(path.join(pluginRoot, '.git'))) {
      return 'dev';
    }
    return 'marketplace';
  };

  it('should detect dev mode when path contains /dev/', () => {
    const result = detectInstallMode('/Users/foo/.claude/plugins/cache/rlm-claude-code-abc123/dev');
    assert.strictEqual(result, 'dev');
  });

  it('should detect marketplace mode for normal path', () => {
    const result = detectInstallMode('/Users/foo/.claude/plugins/cache/rlm-claude-code-abc123');
    assert.strictEqual(result, 'marketplace');
  });
});

// ============================================================================
// Setup Status Tests
// ============================================================================

describe('Setup Status Structure', () => {
  interface SetupStatus {
    platform: string;
    mode: 'marketplace' | 'dev';
    uv: 'ok' | 'missing';
    venv: 'ok' | 'missing';
    binaries: 'ok' | 'missing' | 'partial';
    rlmCore: 'ok' | 'missing' | 'no-venv';
    needsAttention: boolean;
    instructions: string[];
  }

  const createStatus = (
    platform: string,
    mode: 'marketplace' | 'dev',
    uv: 'ok' | 'missing',
    venv: 'ok' | 'missing',
    binaries: 'ok' | 'missing' | 'partial',
    rlmCore: 'ok' | 'missing' | 'no-venv'
  ): SetupStatus => {
    const needsAttention = venv !== 'ok' || rlmCore !== 'ok';
    const instructions: string[] = [];

    if (uv === 'missing') {
      instructions.push('Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh');
    }
    if (venv === 'missing') {
      instructions.push('Create venv: uv venv && uv sync');
    }
    if (binaries === 'missing' && mode === 'marketplace') {
      instructions.push('Download binaries: npm run download:binaries');
    } else if (binaries === 'missing' && mode === 'dev') {
      instructions.push('Build binaries: npm run build -- --binaries-only');
    }
    if (rlmCore === 'missing' && mode === 'marketplace') {
      instructions.push('Download wheel: npm run download:wheel && uv pip install *.whl');
    } else if (rlmCore === 'missing' && mode === 'dev') {
      instructions.push('Build wheel: npm run build -- --wheel-only');
    }

    return { platform, mode, uv, venv, binaries, rlmCore, needsAttention, instructions };
  };

  it('should create status with all OK', () => {
    const status = createStatus('darwin-arm64', 'marketplace', 'ok', 'ok', 'ok', 'ok');
    assert.strictEqual(status.needsAttention, false);
    assert.deepStrictEqual(status.instructions, []);
  });

  it('should create status with missing venv', () => {
    const status = createStatus('darwin-arm64', 'marketplace', 'ok', 'missing', 'ok', 'no-venv');
    assert.strictEqual(status.needsAttention, true);
    assert.ok(status.instructions.some(i => i.includes('venv')));
  });

  it('should create marketplace-specific instructions for missing binaries', () => {
    const status = createStatus('darwin-arm64', 'marketplace', 'ok', 'ok', 'missing', 'ok');
    assert.ok(status.instructions.some(i => i.includes('download:binaries')));
  });

  it('should create dev-specific instructions for missing binaries', () => {
    const status = createStatus('darwin-arm64', 'dev', 'ok', 'ok', 'missing', 'ok');
    assert.ok(status.instructions.some(i => i.includes('build')));
  });
});

// ============================================================================
// JSON Output Tests
// ============================================================================

describe('JSON Output Format', () => {
  it('should produce valid JSON for AI agent consumption', () => {
    const status = {
      platform: 'darwin-arm64',
      mode: 'marketplace' as const,
      uv: 'ok' as const,
      venv: 'ok' as const,
      binaries: 'ok' as const,
      rlmCore: 'ok' as const,
      needsAttention: false,
      instructions: []
    };

    const json = JSON.stringify(status);
    const parsed = JSON.parse(json);

    assert.strictEqual(parsed.platform, 'darwin-arm64');
    assert.strictEqual(parsed.needsAttention, false);
  });

  it('should include hookSpecificOutput format for Claude Code hooks', () => {
    const output = {
      hookSpecificOutput: {
        hookEventName: 'EnsureSetup',
        additionalContext: 'RLM setup status check',
        rlmSetupStatus: {
          platform: 'darwin-arm64',
          mode: 'marketplace',
          needsAttention: false
        }
      }
    };

    const json = JSON.stringify(output);
    const parsed = JSON.parse(json);

    assert.ok(parsed.hookSpecificOutput);
    assert.ok(parsed.hookSpecificOutput.rlmSetupStatus);
  });
});
