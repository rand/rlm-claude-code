#!/usr/bin/env npx ts-node
/**
 * verify-install.ts
 *
 * Post-install verification script.
 * Runs after npm install to confirm setup is complete.
 */

import fs from 'fs';
import path from 'path';
import { ROOT_DIR, log, fileExists, getPlatform } from './common';

function verifyVenv(): boolean {
  log('\n=== Verifying Virtual Environment ===', 'cyan');

  if (fileExists('.venv/bin/activate') || fileExists('.venv/Scripts/activate')) {
    log('Virtual environment exists.', 'green');
    return true;
  }

  log('Virtual environment NOT found.', 'red');
  log('Run: npm install', 'yellow');
  return false;
}

function verifyBinaries(): boolean {
  log('\n=== Verifying Binaries ===', 'cyan');

  const { os, platform } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');

  if (!fs.existsSync(binDir)) {
    log('bin/ directory not found.', 'yellow');
    log('Run: npm run download:binaries', 'yellow');
    return false;
  }

  const binaries = ['session-init', 'complexity-check', 'trajectory-save'];
  let allFound = true;

  for (const binary of binaries) {
    const binaryName = os === 'windows' ? `${binary}-${platform}.exe` : `${binary}-${platform}`;
    const binaryPath = path.join(binDir, binaryName);

    if (fs.existsSync(binaryPath)) {
      log(`  [OK] ${binaryName}`, 'green');
    } else {
      log(`  [MISSING] ${binaryName}`, 'yellow');
      allFound = false;
    }
  }

  return allFound;
}

function verifyPythonDeps(): boolean {
  log('\n=== Verifying Python Dependencies ===', 'cyan');

  if (fileExists('.venv/lib') || fileExists('.venv/Lib')) {
    log('Python packages installed.', 'green');
    return true;
  }

  log('Python dependencies may not be installed.', 'yellow');
  log('Run: uv sync', 'yellow');
  return false;
}

function main(): void {
  log('\n=========================================', 'cyan');
  log('  RLM-Claude-Code Installation Verify', 'cyan');
  log('=========================================', 'cyan');

  const results = {
    venv: verifyVenv(),
    binaries: verifyBinaries(),
    python: verifyPythonDeps(),
  };

  log('\n=== Summary ===', 'cyan');
  log(`Virtual Environment: ${results.venv ? 'OK' : 'NEEDS FIX'}`, results.venv ? 'green' : 'yellow');
  log(`Hook Binaries: ${results.binaries ? 'OK' : 'PARTIAL'}`, results.binaries ? 'green' : 'yellow');
  log(`Python Dependencies: ${results.python ? 'OK' : 'NEEDS FIX'}`, results.python ? 'green' : 'yellow');

  const allOk = results.venv && results.python;

  if (allOk) {
    log('\nInstallation verified successfully!', 'green');
    log('Run: npm run verify  # Full verification', 'cyan');
  } else {
    log('\nInstallation incomplete. Run: npm install', 'yellow');
  }

  process.exit(allOk ? 0 : 1);
}

main();
