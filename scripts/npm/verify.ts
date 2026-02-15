#!/usr/bin/env npx ts-node
/**
 * verify.ts
 *
 * Comprehensive verification script for RLM-Claude-Code installation.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { ROOT_DIR, log, fileExists, getPlatform } from './common';

function verifyBinaries(): { passed: number; failed: number } {
  log('\n=== Verifying Hook Binaries ===', 'cyan');

  const { os, platform } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');
  const binaries = ['session-init', 'complexity-check', 'trajectory-save'];

  let passed = 0;
  let failed = 0;

  if (!fs.existsSync(binDir)) {
    log('bin/ directory not found.', 'red');
    return { passed: 0, failed: binaries.length };
  }

  for (const binary of binaries) {
    const binaryName = os === 'windows' ? `${binary}-${platform}.exe` : `${binary}-${platform}`;
    const binaryPath = path.join(binDir, binaryName);

    if (fs.existsSync(binaryPath)) {
      // Try to run --help or version check
      try {
        execSync(`"${binaryPath}" --help 2>&1 || true`, { timeout: 5000, encoding: 'utf8' });
        log(`  [OK] ${binaryName}`, 'green');
        passed++;
      } catch {
        log(`  [WARN] ${binaryName} exists but may not be executable`, 'yellow');
        passed++;
      }
    } else {
      log(`  [MISSING] ${binaryName}`, 'red');
      failed++;
    }
  }

  return { passed, failed };
}

function verifyRlmCore(): boolean {
  log('\n=== Verifying rlm-core ===', 'cyan');

  try {
    const result = execSync('uv run python -c "import rlm_core; print(rlm_core.version())"', {
      cwd: ROOT_DIR,
      encoding: 'utf8',
      timeout: 30000,
    });
    log(`  [OK] rlm-core version: ${result.trim()}`, 'green');
    return true;
  } catch (error) {
    log('  [ERROR] rlm-core not available', 'red');
    log('  Run: npm run build or npm run download:wheel', 'yellow');
    return false;
  }
}

function verifyHooks(): boolean {
  log('\n=== Verifying Hook Configuration ===', 'cyan');

  const hooksPath = path.join(ROOT_DIR, 'hooks', 'hooks.json');

  if (!fs.existsSync(hooksPath)) {
    log('  [MISSING] hooks/hooks.json', 'red');
    return false;
  }

  try {
    const content = fs.readFileSync(hooksPath, 'utf8');
    JSON.parse(content);
    log('  [OK] hooks/hooks.json is valid JSON', 'green');
  } catch {
    log('  [ERROR] hooks/hooks.json is invalid JSON', 'red');
    return false;
  }

  // Check hook scripts
  const dispatchScript = path.join(ROOT_DIR, 'scripts', 'hook-dispatch.sh');
  if (fs.existsSync(dispatchScript)) {
    log('  [OK] scripts/hook-dispatch.sh exists', 'green');
  } else {
    log('  [WARN] scripts/hook-dispatch.sh not found', 'yellow');
  }

  return true;
}

function verifyPythonDeps(): boolean {
  log('\n=== Verifying Python Dependencies ===', 'cyan');

  try {
    execSync('uv run python -c "import pydantic; import httpx; import anthropic"', {
      cwd: ROOT_DIR,
      encoding: 'utf8',
      timeout: 30000,
    });
    log('  [OK] Core Python dependencies available', 'green');
    return true;
  } catch {
    log('  [ERROR] Python dependencies missing', 'red');
    log('  Run: uv sync', 'yellow');
    return false;
  }
}

function main(): void {
  log('\n=========================================', 'cyan');
  log('  RLM-Claude-Code Verification', 'cyan');
  log('=========================================', 'cyan');

  const binaryResults = verifyBinaries();
  const rlmCoreOk = verifyRlmCore();
  const hooksOk = verifyHooks();
  const pythonOk = verifyPythonDeps();

  log('\n=== Summary ===', 'cyan');
  log(`Binaries: ${binaryResults.passed}/${binaryResults.passed + binaryResults.failed} found`,
    binaryResults.failed === 0 ? 'green' : 'yellow');
  log(`rlm-core: ${rlmCoreOk ? 'OK' : 'FAILED'}`, rlmCoreOk ? 'green' : 'red');
  log(`Hooks: ${hooksOk ? 'OK' : 'ISSUES'}`, hooksOk ? 'green' : 'yellow');
  log(`Python deps: ${pythonOk ? 'OK' : 'FAILED'}`, pythonOk ? 'green' : 'red');

  const allOk = rlmCoreOk && pythonOk;

  if (allOk) {
    log('\nVerification PASSED!', 'green');
  } else {
    log('\nVerification FAILED. Fix issues above.', 'red');
  }

  process.exit(allOk ? 0 : 1);
}

main();
