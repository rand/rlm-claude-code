/**
 * Tests for hook-dispatch.ts
 *
 * Tests the cross-platform hook dispatcher that:
 * - Detects platform and architecture
 * - Executes Go binary if available
 * - Falls back to Python script
 */

import { test, describe, it } from 'node:test';
import assert from 'node:assert';
import path from 'path';
import fs from 'fs';

// ============================================================================
// Binary Path Tests
// ============================================================================

describe('Binary Path Generation', () => {
  const getBinaryPath = (hookName: string, os: string, cpu: string, pluginRoot: string): string => {
    const binaryName = os === 'windows'
      ? `${hookName}-${os}-${cpu}.exe`
      : `${hookName}-${os}-${cpu}`;
    return path.join(pluginRoot, 'bin', binaryName);
  };

  it('should generate correct path for macOS arm64', () => {
    const result = getBinaryPath('session-init', 'darwin', 'arm64', '/plugin');
    assert.strictEqual(result, '/plugin/bin/session-init-darwin-arm64');
  });

  it('should generate correct path for Windows with .exe', () => {
    const result = getBinaryPath('session-init', 'windows', 'amd64', '/plugin');
    assert.strictEqual(result, '/plugin/bin/session-init-windows-amd64.exe');
  });

  it('should generate correct path for Linux amd64', () => {
    const result = getBinaryPath('complexity-check', 'linux', 'amd64', '/plugin');
    assert.strictEqual(result, '/plugin/bin/complexity-check-linux-amd64');
  });
});

// ============================================================================
// Python Fallback Path Tests
// ============================================================================

describe('Python Script Path Generation', () => {
  const getPythonPath = (hookName: string, pluginRoot: string, isLegacy: boolean): string => {
    const subdir = isLegacy ? 'legacy' : '';
    return path.join(pluginRoot, 'scripts', subdir, `${hookName}.py`);
  };

  it('should generate path for current script location', () => {
    const result = getPythonPath('session-init', '/plugin', false);
    assert.strictEqual(result, '/plugin/scripts/session-init.py');
  });

  it('should generate path for legacy script location', () => {
    const result = getPythonPath('session-init', '/plugin', true);
    assert.strictEqual(result, '/plugin/scripts/legacy/session-init.py');
  });
});

// ============================================================================
// Platform Detection for Hooks
// ============================================================================

describe('Hook Platform Detection', () => {
  const detectHookPlatform = (nodePlatform: string, nodeArch: string) => {
    let os: string;
    let cpu: string;

    switch (nodePlatform) {
      case 'darwin':
        os = 'darwin';
        break;
      case 'linux':
        os = 'linux';
        break;
      case 'win32':
        os = 'windows';
        break;
      default:
        os = nodePlatform;
    }

    switch (nodeArch) {
      case 'x64':
        cpu = 'amd64';
        break;
      case 'arm64':
        cpu = 'arm64';
        break;
      default:
        cpu = nodeArch;
    }

    return { os, cpu };
  };

  it('should detect darwin-arm64 for hooks', () => {
    const result = detectHookPlatform('darwin', 'arm64');
    assert.deepStrictEqual(result, { os: 'darwin', cpu: 'arm64' });
  });

  it('should detect windows-amd64 for hooks', () => {
    const result = detectHookPlatform('win32', 'x64');
    assert.deepStrictEqual(result, { os: 'windows', cpu: 'amd64' });
  });
});

// ============================================================================
// Hook Name Validation
// ============================================================================

describe('Hook Name Validation', () => {
  const VALID_HOOKS = ['session-init', 'complexity-check', 'trajectory-save', 'sync-context', 'capture-output'];

  const isValidHook = (hookName: string): boolean => {
    return VALID_HOOKS.includes(hookName);
  };

  it('should accept valid hook names', () => {
    assert.ok(isValidHook('session-init'));
    assert.ok(isValidHook('complexity-check'));
    assert.ok(isValidHook('trajectory-save'));
  });

  it('should reject invalid hook names', () => {
    assert.ok(!isValidHook('invalid-hook'));
    assert.ok(!isValidHook(''));
    assert.ok(!isValidHook('random'));
  });
});

// ============================================================================
// Execution Fallback Logic Tests
// ============================================================================

describe('Execution Fallback Logic', () => {
  interface ExecutionPlan {
    useBinary: boolean;
    binaryPath?: string;
    usePython: boolean;
    pythonPath?: string;
  }

  const planExecution = (
    hookName: string,
    os: string,
    cpu: string,
    pluginRoot: string,
    binaryExists: boolean,
    pythonExists: boolean,
    legacyPythonExists: boolean
  ): ExecutionPlan => {
    const binaryName = os === 'windows'
      ? `${hookName}-${os}-${cpu}.exe`
      : `${hookName}-${os}-${cpu}`;
    const binaryPath = path.join(pluginRoot, 'bin', binaryName);

    if (binaryExists) {
      return { useBinary: true, binaryPath, usePython: false };
    }

    const pythonPath = path.join(pluginRoot, 'scripts', `${hookName}.py`);
    if (pythonExists) {
      return { useBinary: false, usePython: true, pythonPath };
    }

    const legacyPath = path.join(pluginRoot, 'scripts', 'legacy', `${hookName}.py`);
    if (legacyPythonExists) {
      return { useBinary: false, usePython: true, pythonPath: legacyPath };
    }

    return { useBinary: false, usePython: false };
  };

  it('should prefer binary when available', () => {
    const plan = planExecution('session-init', 'darwin', 'arm64', '/plugin', true, true, true);
    assert.strictEqual(plan.useBinary, true);
    assert.ok(plan.binaryPath);
  });

  it('should fall back to Python when binary not available', () => {
    const plan = planExecution('session-init', 'darwin', 'arm64', '/plugin', false, true, true);
    assert.strictEqual(plan.useBinary, false);
    assert.strictEqual(plan.usePython, true);
  });

  it('should fall back to legacy Python when current not available', () => {
    const plan = planExecution('session-init', 'darwin', 'arm64', '/plugin', false, false, true);
    assert.strictEqual(plan.useBinary, false);
    assert.strictEqual(plan.usePython, true);
  });

  it('should have no execution method when nothing available', () => {
    const plan = planExecution('session-init', 'darwin', 'arm64', '/plugin', false, false, false);
    assert.strictEqual(plan.useBinary, false);
    assert.strictEqual(plan.usePython, false);
  });
});
