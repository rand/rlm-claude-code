#!/usr/bin/env npx ts-node
/**
 * ensure-setup.ts
 *
 * Self-healing setup script for rlm-claude-code plugin.
 * Runs on first session start to ensure everything is properly set up.
 *
 * Handles:
 * - Detecting installation mode (marketplace vs dev)
 * - Checking for missing binaries/wheel/venv
 * - Auto-downloading from GitHub releases (marketplace)
 * - Providing build instructions (dev)
 * - Outputting JSON status for AI agent
 *
 * Usage:
 *   ensure-setup.ts           # Check and report status
 *   ensure-setup.ts --fix     # Attempt auto-fix
 *   ensure-setup.ts --json    # Output JSON for hooks
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { ROOT_DIR, log, getPlatform, runCommand, fileExists } from './common';

// Types
type InstallMode = 'marketplace' | 'dev';
type Status = 'ok' | 'missing' | 'partial' | 'no-venv';

interface SetupStatus {
  platform: string;
  mode: InstallMode;
  uv: Status;
  venv: Status;
  binaries: Status;
  rlmCore: Status;
  needsAttention: boolean;
  instructions: string[];
}

// ============================================================================
// Detection Functions
// ============================================================================

/**
 * Detect installation mode
 * - marketplace: installed via `claude plugins install`
 * - dev: symlinked from local dev folder
 */
function detectInstallMode(): InstallMode {
  // Check if path contains /dev/ or ends with /dev
  if (ROOT_DIR.includes('/dev/') || ROOT_DIR.endsWith('/dev')) {
    return 'dev';
  }

  // Check if .git directory exists (indicates dev/clone)
  if (fs.existsSync(path.join(ROOT_DIR, '.git'))) {
    return 'dev';
  }

  return 'marketplace';
}

/**
 * Check if uv is installed
 */
function checkUv(): Status {
  try {
    execSync('uv --version', { stdio: 'pipe' });
    return 'ok';
  } catch {
    return 'missing';
  }
}

/**
 * Check if venv exists
 */
function checkVenv(): Status {
  const venvPath = process.platform === 'win32'
    ? path.join(ROOT_DIR, '.venv', 'Scripts', 'activate')
    : path.join(ROOT_DIR, '.venv', 'bin', 'activate');

  return fs.existsSync(venvPath) ? 'ok' : 'missing';
}

/**
 * Check if binaries exist for current platform
 */
function checkBinaries(): Status {
  const { os, cpu } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');

  if (!fs.existsSync(binDir)) {
    return 'missing';
  }

  const binaries = ['session-init', 'complexity-check', 'trajectory-save'];
  const expected = binaries.map(b =>
    os === 'windows' ? `${b}-${os}-${cpu}.exe` : `${b}-${os}-${cpu}`
  );

  let found = 0;
  for (const binary of expected) {
    if (fs.existsSync(path.join(binDir, binary))) {
      found++;
    }
  }

  if (found === 0) return 'missing';
  if (found < expected.length) return 'partial';
  return 'ok';
}

/**
 * Check for wheel file location
 * Returns: 'root' | 'vendor' | 'none'
 */
function checkWheelLocation(): 'root' | 'vendor' | 'none' {
  // Check root directory first
  if (fs.existsSync(ROOT_DIR)) {
    const rootFiles = fs.readdirSync(ROOT_DIR);
    if (rootFiles.some(f => f.startsWith('rlm_claude_code') && f.endsWith('.whl'))) {
      return 'root';
    }
  }

  // Check vendor directory
  const vendorWheelDir = path.join(ROOT_DIR, 'vendor/loop/rlm-core/target/wheels');
  if (fs.existsSync(vendorWheelDir)) {
    const vendorFiles = fs.readdirSync(vendorWheelDir);
    if (vendorFiles.some(f => f.startsWith('rlm_claude_code') && f.endsWith('.whl'))) {
      return 'vendor';
    }
  }

  return 'none';
}

/**
 * Check if rlm_core is installed
 */
function checkRlmCore(): Status {
  const venvStatus = checkVenv();
  if (venvStatus !== 'ok') {
    return 'no-venv';
  }

  const venvPython = process.platform === 'win32'
    ? path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe')
    : path.join(ROOT_DIR, '.venv', 'bin', 'python3');

  if (!fs.existsSync(venvPython)) {
    return 'no-venv';
  }

  try {
    execSync(`"${venvPython}" -c "import rlm_core"`, { stdio: 'pipe' });
    return 'ok';
  } catch {
    return 'missing';
  }
}

// ============================================================================
// Fix Functions
// ============================================================================

/**
 * Attempt to fix missing venv
 */
function fixVenv(): boolean {
  if (checkUv() !== 'ok') {
    log('uv is not installed. Cannot create venv.', 'red');
    return false;
  }

  try {
    log('Creating venv with uv...', 'yellow');
    runCommand('uv venv', { silent: true });
    runCommand('uv sync', { silent: true });
    log('  [OK] venv created', 'green');
    return true;
  } catch (error) {
    log(`  [FAIL] Could not create venv: ${(error as Error).message}`, 'red');
    return false;
  }
}

/**
 * Attempt to fix missing binaries
 */
function fixBinaries(mode: InstallMode): boolean {
  if (mode === 'dev') {
    log('Dev mode: Skipping binary download. Build from source instead.', 'yellow');
    log('  Run: npm run build -- --binaries-only', 'cyan');
    return false;
  }

  try {
    log('Downloading pre-built binaries...', 'yellow');
    const scriptPath = path.join(ROOT_DIR, 'scripts/npm/download-binaries.ts');
    runCommand(`npx ts-node "${scriptPath}"`, { silent: false });
    return true;
  } catch (error) {
    log(`  [FAIL] Could not download binaries: ${(error as Error).message}`, 'red');
    return false;
  }
}

/**
 * Attempt to fix missing rlm_core
 */
function fixRlmCore(mode: InstallMode): boolean {
  if (checkVenv() !== 'ok') {
    log('venv not found. Creating venv first...', 'yellow');
    if (!fixVenv()) {
      return false;
    }
  }

  // Check for wheel files in various locations
  // Priority: root (for marketplace) > vendor (for dev builds) > target
  const wheelLocations = [
    ROOT_DIR,  // Root directory (for marketplace packages with bundled wheel)
    path.join(ROOT_DIR, 'vendor/loop/rlm-core/target/wheels'),
    path.join(ROOT_DIR, 'target/wheels'),
  ];

  for (const location of wheelLocations) {
    if (!fs.existsSync(location)) continue;

    const isDir = fs.statSync(location).isDirectory();
    const files = isDir ? fs.readdirSync(location) : [path.basename(location)];

    for (const file of files) {
      if (file.startsWith('rlm_claude_code') && file.endsWith('.whl')) {
        const wheelPath = isDir ? path.join(location, file) : location;
        log(`Found wheel: ${file}`, 'green');

        try {
          runCommand(`uv pip install "${wheelPath}"`, { silent: false });
          log('  [OK] Installed wheel', 'green');
          return true;
        } catch (error) {
          log(`  [FAIL] Could not install wheel: ${(error as Error).message}`, 'red');
        }
      }
    }
  }

  if (mode === 'dev') {
    log('Dev mode: No local wheel found. Build from source instead.', 'yellow');
    log('  Run: npm run build -- --wheel-only', 'cyan');
    return false;
  }

  try {
    log('Downloading pre-built wheel...', 'yellow');
    const scriptPath = path.join(ROOT_DIR, 'scripts/npm/download-wheel.ts');
    runCommand(`npx ts-node "${scriptPath}"`, { silent: false });

    // Find and install the downloaded wheel
    const files = fs.readdirSync(ROOT_DIR);
    const wheel = files.find(f => f.startsWith('rlm_claude_code') && f.endsWith('.whl'));

    if (wheel) {
      log(`Installing wheel: ${wheel}`, 'yellow');
      runCommand(`uv pip install ${wheel}`, { silent: false });
      // Clean up wheel file
      fs.unlinkSync(path.join(ROOT_DIR, wheel));
    }

    return true;
  } catch (error) {
    log(`  [FAIL] Could not install rlm_core: ${(error as Error).message}`, 'red');
    return false;
  }
}

// ============================================================================
// Main Logic
// ============================================================================

function createStatus(
  platform: string,
  mode: InstallMode,
  uv: Status,
  venv: Status,
  binaries: Status,
  rlmCore: Status
): SetupStatus {
  const needsAttention = venv !== 'ok' || rlmCore !== 'ok';
  const instructions: string[] = [];

  if (uv === 'missing') {
    instructions.push('Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh');
  }
  if (venv === 'missing') {
    instructions.push('Create venv: uv venv && uv sync');
  }
  if (binaries === 'missing' || binaries === 'partial') {
    if (mode === 'marketplace') {
      instructions.push('Download binaries: npm run download:binaries');
    } else {
      instructions.push('Build binaries: npm run build -- --binaries-only');
    }
  }
  if (rlmCore === 'missing' || rlmCore === 'no-venv') {
    if (mode === 'marketplace') {
      instructions.push('Download and install wheel: npm run download:wheel && uv pip install *.whl');
    } else {
      instructions.push('Build wheel: npm run build -- --wheel-only');
    }
  }

  return { platform, mode, uv, venv, binaries, rlmCore, needsAttention, instructions };
}

function outputJson(status: SetupStatus): void {
  // Build AI-friendly context
  let additionalContext = 'RLM plugin setup status: ';
  if (status.needsAttention) {
    additionalContext += 'NEEDS ATTENTION. ';
    if (status.uv === 'missing') {
      additionalContext += 'uv package manager is not installed. ';
    }
    if (status.venv === 'missing') {
      additionalContext += 'Python virtual environment is not set up. ';
    }
    if (status.rlmCore === 'missing' || status.rlmCore === 'no-venv') {
      additionalContext += 'rlm_core Python package is not installed. ';
    }
    additionalContext += 'Please help the user by running the suggested commands in the instructions array.';
  } else {
    additionalContext += 'All checks passed. Plugin is ready to use.';
  }

  const output = {
    hookSpecificOutput: {
      hookEventName: 'EnsureSetup',
      additionalContext,
      rlmSetupStatus: status
    }
  };

  console.log(JSON.stringify(output));
}

function outputText(status: SetupStatus): void {
  log('\n=========================================', 'cyan');
  log('  RLM Plugin Setup Status', 'cyan');
  log('=========================================', 'cyan');

  log(`\nPlatform: ${status.platform}`, 'dim');
  log(`Mode:     ${status.mode}`, 'dim');

  log('\nStatus:', 'yellow');
  log(`  uv:       ${status.uv}`, status.uv === 'ok' ? 'green' : 'red');
  log(`  venv:     ${status.venv}`, status.venv === 'ok' ? 'green' : 'red');
  log(`  binaries: ${status.binaries}`, status.binaries === 'ok' ? 'green' : 'yellow');
  log(`  rlm_core: ${status.rlmCore}`, status.rlmCore === 'ok' ? 'green' : 'red');

  if (status.needsAttention) {
    log('\nAction Required:', 'yellow');
    for (const instruction of status.instructions) {
      log(`  • ${instruction}`, 'cyan');
    }
  } else {
    log('\n✓ All checks passed!', 'green');
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const shouldFix = args.includes('--fix');
  const jsonOutput = args.includes('--json');

  // Detect platform and mode
  const { platform } = getPlatform();
  const mode = detectInstallMode();

  // Check status
  let uv = checkUv();
  let venv = checkVenv();
  let binaries = checkBinaries();
  let rlmCore = checkRlmCore();

  // Attempt fixes if requested
  if (shouldFix) {
    log('\n=== Attempting Auto-Fix ===\n', 'cyan');

    if (venv !== 'ok') {
      fixVenv();
    }

    if (binaries !== 'ok') {
      fixBinaries(mode);
    }

    if (rlmCore !== 'ok') {
      fixRlmCore(mode);
    }

    // Re-check after fixes
    venv = checkVenv();
    binaries = checkBinaries();
    rlmCore = checkRlmCore();
  }

  // Create status
  const status = createStatus(platform, mode, uv, venv, binaries, rlmCore);

  // Output
  if (jsonOutput) {
    outputJson(status);
  } else {
    outputText(status);
  }

  // Exit with appropriate code
  process.exit(status.needsAttention ? 1 : 0);
}

main().catch((error) => {
  console.error(`Setup check failed: ${(error as Error).message}`);
  process.exit(1);
});
