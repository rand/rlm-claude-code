#!/usr/bin/env npx ts-node
/**
 * clean.ts
 *
 * Remove build artifacts from RLM-Claude-Code project.
 *
 * Usage:
 *   npm run clean          # Remove build artifacts
 *   npm run clean -- --deep  # Also remove .venv
 */

import fs from 'fs';
import path from 'path';
import { ROOT_DIR, log, fileExists } from './common';

interface CleanOptions {
  deep: boolean;
}

function parseArgs(): CleanOptions {
  return {
    deep: process.argv.includes('--deep'),
  };
}

function removeDir(dirPath: string): boolean {
  if (!fs.existsSync(dirPath)) {
    return false;
  }

  try {
    fs.rmSync(dirPath, { recursive: true, force: true });
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Failed to remove ${dirPath}: ${err.message}`, 'red');
    return false;
  }
}

function removeFiles(pattern: string): number {
  const dir = path.dirname(pattern);
  const base = path.basename(pattern);

  if (!fs.existsSync(path.join(ROOT_DIR, dir))) {
    return 0;
  }

  const files = fs.readdirSync(path.join(ROOT_DIR, dir));
  let removed = 0;

  for (const file of files) {
    if (file.startsWith(base.replace('*', '')) || file === base) {
      const filePath = path.join(ROOT_DIR, dir, file);
      try {
        fs.rmSync(filePath, { force: true });
        removed++;
      } catch {
        // ignore
      }
    }
  }

  return removed;
}

function main(): void {
  log('\n=========================================', 'cyan');
  log('  RLM-Claude-Code Clean', 'cyan');
  log('=========================================', 'cyan');

  const options = parseArgs();

  log('\n=== Removing Build Artifacts ===', 'cyan');

  // Remove wheel files
  const wheelsRemoved = removeFiles('*.whl');
  if (wheelsRemoved > 0) {
    log(`Removed ${wheelsRemoved} wheel file(s)`, 'green');
  }

  // Remove __pycache__
  if (fileExists('src/__pycache__')) {
    removeDir(path.join(ROOT_DIR, 'src/__pycache__'));
    log('Removed src/__pycache__', 'green');
  }

  // Remove .pytest_cache
  if (fileExists('.pytest_cache')) {
    removeDir(path.join(ROOT_DIR, '.pytest_cache'));
    log('Removed .pytest_cache', 'green');
  }

  // Remove .ruff_cache
  if (fileExists('.ruff_cache')) {
    removeDir(path.join(ROOT_DIR, '.ruff_cache'));
    log('Removed .ruff_cache', 'green');
  }

  // Remove dist/
  if (fileExists('dist')) {
    removeDir(path.join(ROOT_DIR, 'dist'));
    log('Removed dist/', 'green');
  }

  // Remove build/
  if (fileExists('build')) {
    removeDir(path.join(ROOT_DIR, 'build'));
    log('Removed build/', 'green');
  }

  // Remove .mypy_cache
  if (fileExists('.mypy_cache')) {
    removeDir(path.join(ROOT_DIR, '.mypy_cache'));
    log('Removed .mypy_cache', 'green');
  }

  // Remove prerequisites check file
  if (fileExists('.prerequisites-check.json')) {
    fs.rmSync(path.join(ROOT_DIR, '.prerequisites-check.json'), { force: true });
    log('Removed .prerequisites-check.json', 'green');
  }

  // Deep clean: remove .venv
  if (options.deep) {
    log('\n=== Deep Clean ===', 'cyan');
    if (fileExists('.venv')) {
      removeDir(path.join(ROOT_DIR, '.venv'));
      log('Removed .venv/', 'green');
    }
  }

  log('\n=========================================', 'green');
  log('  Clean Complete!', 'green');
  log('=========================================', 'green');
}

main();
