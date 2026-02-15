#!/usr/bin/env npx ts-node
/**
 * build.ts
 *
 * Build script for RLM-Claude-Code.
 * Supports both downloading pre-built binaries and building from source.
 *
 * Usage:
 *   npm run build              # Download pre-built binaries (default)
 *   npm run build -- --all     # Build everything from source
 *   npm run build -- --binaries-only  # Build only Go binaries
 *   npm run build -- --wheel-only     # Build only rlm-core wheel
 */

import fs from 'fs';
import path from 'path';
import { ROOT_DIR, log, runCommand, fileExists, getPlatform } from './common';

interface BuildOptions {
  all: boolean;
  binariesOnly: boolean;
  wheelOnly: boolean;
}

function parseArgs(): BuildOptions {
  return {
    all: process.argv.includes('--all'),
    binariesOnly: process.argv.includes('--binaries-only'),
    wheelOnly: process.argv.includes('--wheel-only'),
  };
}

function buildWheelFromSource(): boolean {
  log('\n=== Building rlm-core Wheel from Source ===', 'cyan');

  if (!fileExists('vendor/loop/rlm-core/Cargo.toml')) {
    log('rlm-core submodule not found. Run: git submodule update --init --recursive', 'red');
    return false;
  }

  try {
    log('Building rlm-core with maturin...');
    runCommand('uv run maturin develop --release -v', { cwd: path.join(ROOT_DIR, 'vendor/loop/rlm-core') });
    log('rlm-core wheel built successfully.', 'green');

    // Copy wheel to root for marketplace distribution
    const wheelDir = path.join(ROOT_DIR, 'vendor/loop/rlm-core/target/wheels');
    if (fs.existsSync(wheelDir)) {
      const wheels = fs.readdirSync(wheelDir).filter(f => f.endsWith('.whl'));
      if (wheels.length > 0) {
        const srcWheel = path.join(wheelDir, wheels[0]);
        const destWheel = path.join(ROOT_DIR, wheels[0]);
        fs.copyFileSync(srcWheel, destWheel);
        log(`Copied wheel to root: ${wheels[0]}`, 'dim');
      }
    }

    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error building wheel: ${err.message}`, 'red');
    return false;
  }
}

function buildGoBinaries(): boolean {
  log('\n=== Building Go Hook Binaries ===', 'cyan');

  const { platform } = getPlatform();

  if (!fileExists('Makefile')) {
    log('Makefile not found.', 'red');
    return false;
  }

  try {
    log('Building Go binaries with Make...');
    runCommand(`make PLATFORM=${platform}`);
    log('Go binaries built successfully.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error building binaries: ${err.message}`, 'red');
    return false;
  }
}

async function downloadBinaries(): Promise<boolean> {
  log('\n=== Downloading Pre-built Binaries ===', 'cyan');

  const { os, platform } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');

  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }

  const binaries = ['session-init', 'complexity-check', 'trajectory-save'];
  const expectedBinary = os === 'windows'
    ? `${binaries[0]}-${platform}.exe`
    : `${binaries[0]}-${platform}`;

  if (fs.existsSync(path.join(binDir, expectedBinary))) {
    log(`Binaries already exist for ${platform}.`, 'green');
    return true;
  }

  log('Downloading binaries...');
  runCommand('npx ts-node scripts/npm/download-binaries.ts', { silent: false });
  return true;
}

async function downloadWheel(): Promise<boolean> {
  log('\n=== Downloading Pre-built Wheel ===', 'cyan');
  runCommand('npx ts-node scripts/npm/download-wheel.ts', { silent: false });
  return true;
}

function installDeps(): boolean {
  log('\n=== Installing Dependencies ===', 'cyan');

  try {
    runCommand('uv sync');
    log('Dependencies installed.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error installing dependencies: ${err.message}`, 'red');
    return false;
  }
}

async function main(): Promise<void> {
  log('\n=========================================', 'cyan');
  log('  RLM-Claude-Code Build', 'cyan');
  log('=========================================', 'cyan');

  const options = parseArgs();

  if (options.all) {
    log('Building everything from source...', 'cyan');

    if (!buildWheelFromSource()) {
      process.exit(1);
    }

    if (!buildGoBinaries()) {
      process.exit(1);
    }

    if (!installDeps()) {
      process.exit(1);
    }
  } else if (options.binariesOnly) {
    if (!buildGoBinaries()) {
      process.exit(1);
    }
  } else if (options.wheelOnly) {
    if (!buildWheelFromSource()) {
      process.exit(1);
    }
  } else {
    // Default: download pre-built
    await downloadBinaries();
    await downloadWheel();
    installDeps();
  }

  log('\n=========================================', 'green');
  log('  Build Complete!', 'green');
  log('=========================================', 'green');
  log('\nNext: npm run verify');
}

main().catch((error) => {
  const err = error as Error;
  log(`\nBuild failed: ${err.message}`, 'red');
  process.exit(1);
});
