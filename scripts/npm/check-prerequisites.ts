#!/usr/bin/env npx ts-node
/**
 * check-prerequisites.ts
 *
 * Verifies that required dependencies are installed before proceeding with setup.
 * Checks: uv, python (3.12+), and git submodules
 */

import fs from 'fs';
import path from 'path';
import { ROOT_DIR, log, checkCommand, type CommandCheckResult, type ColorStyle } from './common';

interface CheckResult {
  name: string;
  status: 'ok' | 'missing' | 'error' | 'needs_init';
  message?: string;
}

interface SubmoduleCheckResult {
  success: boolean;
  note?: string;
  fix?: string;
}

const MIN_PYTHON_MINOR = 12;

function checkPython(): CommandCheckResult {
  const pythonCommands = ['python3', 'python'];

  for (const cmd of pythonCommands) {
    const result = checkCommand(cmd, '--version');
    if (result.success && result.version) {
      const match = result.version.match(/Python\s+(\d+)\.(\d+)/i);
      if (match) {
        const major = parseInt(match[1], 10);
        const minor = parseInt(match[2], 10);
        if (major === 3 && minor >= MIN_PYTHON_MINOR) {
          return { success: true, command: cmd, version: result.version.trim() };
        } else if (major === 3) {
          return {
            success: false,
            command: cmd,
            version: result.version.trim(),
            error: `Python 3.${MIN_PYTHON_MINOR}+ required, found ${major}.${minor}`,
          };
        }
      }
    }
  }

  return { success: false, error: 'Python 3.12+ not found' };
}

function checkUv(): CommandCheckResult {
  return checkCommand('uv', '--version');
}

function checkGitSubmodules(): SubmoduleCheckResult {
  const submodulePath = path.join(ROOT_DIR, 'vendor', 'loop', 'rlm-core');
  const gitmodulesPath = path.join(ROOT_DIR, '.gitmodules');

  if (!fs.existsSync(gitmodulesPath)) {
    return { success: true, note: 'No submodules configured' };
  }

  if (fs.existsSync(submodulePath)) {
    const cargoPath = path.join(submodulePath, 'Cargo.toml');
    if (fs.existsSync(cargoPath)) {
      return { success: true, note: 'Submodules initialized' };
    }
  }

  return {
    success: false,
    note: 'Git submodules need initialization',
    fix: 'Run: git submodule update --init --recursive',
  };
}

interface PrerequisitesResult {
  timestamp: string;
  checks: CheckResult[];
  canProceed: boolean;
}

function main(): void {
  log('\n=== RLM-Claude-Code Prerequisites Check ===\n', 'yellow');

  const checks: CheckResult[] = [];
  let hasErrors = false;

  // Check uv
  log('Checking uv package manager...', 'yellow');
  const uvCheck = checkUv();
  if (uvCheck.success && uvCheck.version) {
    log(`  [OK] uv: ${uvCheck.version}`, 'green');
    checks.push({ name: 'uv', status: 'ok' });
  } else {
    log(`  [MISSING] uv: Not found`, 'red');
    log(`  Install: curl -LsSf https://astral.sh/uv/install.sh | sh`, 'yellow');
    checks.push({ name: 'uv', status: 'missing' });
    hasErrors = true;
  }

  // Check Python
  log('\nChecking Python...', 'yellow');
  const pythonCheck = checkPython();
  if (pythonCheck.success && pythonCheck.version) {
    log(`  [OK] ${pythonCheck.command}: ${pythonCheck.version}`, 'green');
    checks.push({ name: 'python', status: 'ok' });
  } else {
    log(`  [ERROR] ${pythonCheck.error}`, 'red');
    log(`  Install: brew install python@3.12 or visit python.org`, 'yellow');
    checks.push({ name: 'python', status: 'error', message: pythonCheck.error });
    hasErrors = true;
  }

  // Check git submodules
  log('\nChecking git submodules...', 'yellow');
  const submoduleCheck = checkGitSubmodules();
  if (submoduleCheck.success) {
    log(`  [OK] ${submoduleCheck.note}`, 'green');
    checks.push({ name: 'submodules', status: 'ok' });
  } else {
    log(`  [NEEDS FIX] ${submoduleCheck.note}`, 'yellow');
    if (submoduleCheck.fix) {
      log(`  ${submoduleCheck.fix}`, 'yellow');
    }
    checks.push({ name: 'submodules', status: 'needs_init' });
  }

  log('\n=========================================\n');

  const result: PrerequisitesResult = {
    timestamp: new Date().toISOString(),
    checks,
    canProceed: !hasErrors,
  };

  const resultPath = path.join(ROOT_DIR, '.prerequisites-check.json');
  fs.writeFileSync(resultPath, JSON.stringify(result, null, 2));

  if (hasErrors) {
    log('Prerequisites check FAILED. Please install missing dependencies.', 'red');
    process.exit(1);
  } else {
    log('Prerequisites check PASSED.', 'green');
    process.exit(0);
  }
}

main();
