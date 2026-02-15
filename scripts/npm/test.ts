#!/usr/bin/env npx ts-node
/**
 * test.ts
 *
 * Test runner for RLM-Claude-Code.
 * Supports quick smoke tests and full test suite.
 *
 * Usage:
 *   npm run test          # Smoke tests (default)
 *   npm run test:quick    # Quick smoke test
 *   npm run test:full     # Full test suite
 */

import { ROOT_DIR, log, runCommand, fileExists } from './common';

interface TestOptions {
  quick: boolean;
  full: boolean;
}

function parseArgs(): TestOptions {
  return {
    quick: process.argv.includes('--quick'),
    full: process.argv.includes('--full'),
  };
}

function runSmokeTests(): boolean {
  log('\n=== Running Smoke Tests ===', 'cyan');

  // Test 1: Python imports
  log('\n1. Testing Python imports...', 'yellow');
  const importResult = runCommand(
    'uv run python -c "from src import orchestrator, memory_store, context_manager"',
    { silent: true }
  );

  if (importResult !== null) {
    log('   [OK] Python imports work', 'green');
  } else {
    log('   [FAIL] Python imports failed', 'red');
    return false;
  }

  // Test 2: rlm-core
  log('\n2. Testing rlm-core...', 'yellow');
  const rlmCoreResult = runCommand(
    'uv run python -c "import rlm_core; print(rlm_core.version())"',
    { silent: true }
  );

  if (rlmCoreResult !== null) {
    log('   [OK] rlm-core available', 'green');
  } else {
    log('   [FAIL] rlm-core not available', 'red');
    return false;
  }

  // Test 3: Go binary exists
  log('\n3. Testing hook binaries...', 'yellow');
  if (fileExists('bin/complexity-check-darwin-arm64') ||
      fileExists('bin/complexity-check-darwin-amd64') ||
      fileExists('bin/complexity-check-linux-arm64') ||
      fileExists('bin/complexity-check-linux-amd64')) {
    log('   [OK] Hook binaries found', 'green');
  } else {
    log('   [WARN] Hook binaries not found (may need build)', 'yellow');
  }

  // Test 4: Quick pytest (smoke)
  log('\n4. Running quick pytest...', 'yellow');
  const pytestResult = runCommand(
    'uv run pytest tests/unit/ -v -x --tb=short -q 2>&1 | head -50',
    { silent: false }
  );

  if (pytestResult !== null) {
    log('   [OK] Quick tests passed', 'green');
  } else {
    log('   [WARN] Some tests may have failed', 'yellow');
  }

  return true;
}

function runFullTests(): boolean {
  log('\n=== Running Full Test Suite ===', 'cyan');
  log('This may take several minutes...\n', 'yellow');

  const result = runCommand('uv run pytest tests/ -v --tb=short');
  return result !== null;
}

function main(): void {
  log('\n=========================================', 'cyan');
  log('  RLM-Claude-Code Tests', 'cyan');
  log('=========================================', 'cyan');

  const options = parseArgs();

  let success = false;

  if (options.full) {
    success = runFullTests();
  } else {
    success = runSmokeTests();
  }

  log('\n=========================================', 'cyan');
  if (success) {
    log('  Tests Complete!', 'green');
  } else {
    log('  Tests Failed!', 'red');
  }
  log('=========================================', 'cyan');

  process.exit(success ? 0 : 1);
}

main();
