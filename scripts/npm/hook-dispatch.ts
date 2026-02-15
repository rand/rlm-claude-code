#!/usr/bin/env npx ts-node
/**
 * hook-dispatch.ts
 *
 * Cross-platform hook dispatcher for rlm-claude-code plugin.
 * Replaces hook-dispatch.sh for Windows compatibility.
 *
 * Usage:
 *   hook-dispatch.ts <hook-name>
 *
 * Priority:
 * 1. Go binary (bin/<hook>-<os>-<arch>[.exe])
 * 2. Python script (scripts/<hook>.py)
 * 3. Legacy Python script (scripts/legacy/<hook>.py)
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { ROOT_DIR } from './common';

// Valid hook names
const VALID_HOOKS = [
  'session-init',
  'complexity-check',
  'trajectory-save',
  'sync-context',
  'capture-output'
];

// ============================================================================
// Platform Detection
// ============================================================================

interface PlatformInfo {
  os: string;
  cpu: string;
}

function detectPlatform(): PlatformInfo {
  let os: string;
  let cpu: string;

  switch (process.platform) {
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
      os = process.platform;
  }

  switch (process.arch) {
    case 'x64':
      cpu = 'amd64';
      break;
    case 'arm64':
      cpu = 'arm64';
      break;
    default:
      cpu = process.arch;
  }

  return { os, cpu };
}

// ============================================================================
// Path Generation
// ============================================================================

function getBinaryPath(hookName: string, os: string, cpu: string): string {
  const binaryName = os === 'windows'
    ? `${hookName}-${os}-${cpu}.exe`
    : `${hookName}-${os}-${cpu}`;
  return path.join(ROOT_DIR, 'bin', binaryName);
}

function getPythonPath(hookName: string, legacy: boolean = false): string {
  const subdir = legacy ? 'legacy' : '';
  return path.join(ROOT_DIR, 'scripts', subdir, `${hookName}.py`);
}

function getVenvPython(): string {
  return process.platform === 'win32'
    ? path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe')
    : path.join(ROOT_DIR, '.venv', 'bin', 'python3');
}

// ============================================================================
// Execution Functions
// ============================================================================

interface ExecutionResult {
  success: boolean;
  exitCode: number;
  error?: string;
}

/**
 * Execute a command and return result
 */
function executeCommand(command: string, args: string[] = []): Promise<ExecutionResult> {
  return new Promise((resolve) => {
    const proc = spawn(command, args, {
      stdio: 'inherit',
      env: {
        ...process.env,
        CLAUDE_PLUGIN_ROOT: ROOT_DIR
      }
    });

    proc.on('close', (code) => {
      resolve({
        success: code === 0,
        exitCode: code ?? 1
      });
    });

    proc.on('error', (err) => {
      resolve({
        success: false,
        exitCode: 1,
        error: err.message
      });
    });
  });
}

/**
 * Execute Go binary
 */
async function executeBinary(binaryPath: string): Promise<ExecutionResult> {
  if (process.env.HOOK_DEBUG === '1') {
    console.error(`[DEBUG] Executing binary: ${binaryPath}`);
  }
  return executeCommand(binaryPath);
}

/**
 * Execute Python script
 */
async function executePython(scriptPath: string): Promise<ExecutionResult> {
  const venvPython = getVenvPython();
  const python = fs.existsSync(venvPython) ? venvPython : 'python3';

  if (process.env.HOOK_DEBUG === '1') {
    console.error(`[DEBUG] Executing Python: ${python} ${scriptPath}`);
  }
  return executeCommand(python, [scriptPath]);
}

// ============================================================================
// Main Dispatcher
// ============================================================================

async function dispatch(hookName: string): Promise<number> {
  // Validate hook name
  if (!VALID_HOOKS.includes(hookName)) {
    console.error(`Invalid hook: ${hookName}`);
    console.error(`Valid hooks: ${VALID_HOOKS.join(', ')}`);
    return 1;
  }

  const { os, cpu } = detectPlatform();

  if (process.env.HOOK_DEBUG === '1') {
    console.error(`[DEBUG] Platform: ${os}-${cpu}`);
    console.error(`[DEBUG] Plugin root: ${ROOT_DIR}`);
  }

  // Try Go binary first
  const binaryPath = getBinaryPath(hookName, os, cpu);
  if (fs.existsSync(binaryPath)) {
    const result = await executeBinary(binaryPath);
    if (result.success) {
      return 0;
    }
    // If binary fails, fall through to Python
    if (process.env.HOOK_DEBUG === '1') {
      console.error(`[DEBUG] Binary failed, falling back to Python`);
    }
  }

  // Try Python script in current location
  const pythonPath = getPythonPath(hookName);
  if (fs.existsSync(pythonPath)) {
    const result = await executePython(pythonPath);
    if (result.success) {
      return 0;
    }
  }

  // Try legacy Python script
  const legacyPath = getPythonPath(hookName, true);
  if (fs.existsSync(legacyPath)) {
    const result = await executePython(legacyPath);
    if (result.success) {
      return 0;
    }
  }

  // Nothing found
  console.error(`Hook not found: ${hookName}`);
  console.error(`Searched:`);
  console.error(`  - ${binaryPath}`);
  console.error(`  - ${pythonPath}`);
  console.error(`  - ${legacyPath}`);
  return 1;
}

// ============================================================================
// Entry Point
// ============================================================================

async function main(): Promise<void> {
  const hookName = process.argv[2];

  if (!hookName) {
    console.error('Usage: hook-dispatch.ts <hook-name>');
    console.error(`Valid hooks: ${VALID_HOOKS.join(', ')}`);
    process.exit(1);
  }

  const exitCode = await dispatch(hookName);
  process.exit(exitCode);
}

main().catch((error) => {
  console.error(`Hook dispatch failed: ${(error as Error).message}`);
  process.exit(1);
});
