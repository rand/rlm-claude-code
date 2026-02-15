#!/usr/bin/env npx ts-node
/**
 * setup.ts
 *
 * Main installation script for RLM-Claude-Code.
 * Handles:
 * - Git submodule initialization
 * - Virtual environment creation with uv
 * - Downloading pre-built binaries (light users)
 * - Downloading and installing wheel files
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { ROOT_DIR, log, runCommand, fileExists, getPlatform, type RunCommandOptions } from './common';

interface GitHubRelease {
  tag_name: string;
  assets: Array<{
    name: string;
    browser_download_url: string;
  }>;
}

function execSyncSafe(command: string, options: RunCommandOptions = {}): string | null {
  return runCommand(command, options);
}

function initGitSubmodules(): boolean {
  log('\n=== Initializing Git Submodules ===', 'cyan');

  if (!fileExists('.gitmodules')) {
    log('No .gitmodules found, skipping submodule initialization.', 'yellow');
    return true;
  }

  const submodulePath = 'vendor/loop/rlm-core/Cargo.toml';
  if (fileExists(submodulePath)) {
    log('Git submodules already initialized.', 'green');
    return true;
  }

  try {
    log('Initializing git submodules...');
    execSyncSafe('git submodule update --init --recursive');
    log('Git submodules initialized successfully.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Warning: Could not initialize submodules: ${err.message}`, 'yellow');
    log('Continuing without submodules (may affect building from source).', 'yellow');
    return false;
  }
}

function createVenv(): boolean {
  log('\n=== Creating Virtual Environment ===', 'cyan');

  if (fileExists('.venv/bin/activate') || fileExists('.venv/Scripts/activate')) {
    log('Virtual environment already exists.', 'green');
    return true;
  }

  try {
    log('Creating venv with uv...');
    execSyncSafe('uv venv');
    log('Virtual environment created successfully.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error creating venv: ${err.message}`, 'red');
    return false;
  }
}

function downloadFile(url: string, destPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    const file = fs.createWriteStream(destPath);

    protocol.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          downloadFile(redirectUrl, destPath).then(resolve).catch(reject);
          return;
        }
      }

      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode}: ${url}`));
        return;
      }

      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(destPath, () => {});
      reject(err);
    });
  });
}

async function getLatestRelease(): Promise<GitHubRelease> {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.github.com',
      path: '/repos/rand/rlm-claude-code/releases/latest',
      headers: {
        'User-Agent': 'rlm-claude-code-installer',
      },
    };

    https.get(options, (response) => {
      let data = '';
      response.on('data', (chunk) => (data += chunk));
      response.on('end', () => {
        try {
          const release = JSON.parse(data) as GitHubRelease;
          resolve(release);
        } catch (e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
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

  try {
    log('Fetching latest release information...');
    const release = await getLatestRelease();
    const version = release.tag_name;
    log(`Latest version: ${version}`);

    for (const binary of binaries) {
      const binaryName = os === 'windows'
        ? `${binary}-${platform}.exe`
        : `${binary}-${platform}`;
      const asset = release.assets.find((a) => a.name === binaryName);

      if (!asset) {
        log(`Warning: Binary not found for ${binaryName}`, 'yellow');
        log(`Available platforms may not include ${platform}`, 'yellow');
        continue;
      }

      const destPath = path.join(binDir, binaryName);
      log(`Downloading ${binaryName}...`);

      await downloadFile(asset.browser_download_url, destPath);

      if (os !== 'windows') {
        fs.chmodSync(destPath, 0o755);
      }

      log(`  Downloaded: ${binaryName}`, 'green');
    }

    log('All binaries downloaded successfully.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Warning: Could not download binaries: ${err.message}`, 'yellow');
    log('You may need to build from source. Run: npm run build', 'yellow');
    return false;
  }
}

async function downloadWheel(): Promise<string | false> {
  log('\n=== Downloading Python Wheel ===', 'cyan');

  const { os, cpu } = getPlatform();

  let wheelPlatform: string;
  if (os === 'darwin' && cpu === 'arm64') {
    wheelPlatform = 'macosx_11_0_arm64';
  } else if (os === 'darwin' && cpu === 'amd64') {
    wheelPlatform = 'macosx_10_9_x86_64';
  } else if (os === 'linux' && cpu === 'amd64') {
    wheelPlatform = 'manylinux_2_17_x86_64';
  } else if (os === 'linux' && cpu === 'arm64') {
    wheelPlatform = 'manylinux_2_17_aarch64';
  } else {
    wheelPlatform = 'win_amd64';
  }

  try {
    log('Fetching latest release information...');
    const release = await getLatestRelease();

    const wheelAsset = release.assets.find((a) => {
      return (
        a.name.includes('rlm_claude_code') &&
        a.name.endsWith('.whl') &&
        (a.name.includes(wheelPlatform) || a.name.includes('py3-none-any'))
      );
    });

    if (!wheelAsset) {
      log('No pre-built wheel found for your platform.', 'yellow');
      log('Will install from source using uv sync.', 'yellow');
      return false;
    }

    const wheelPath = path.join(ROOT_DIR, wheelAsset.name);
    log(`Downloading ${wheelAsset.name}...`);
    await downloadFile(wheelAsset.browser_download_url, wheelPath);
    log(`Wheel downloaded: ${wheelAsset.name}`, 'green');

    return wheelAsset.name;
  } catch (error) {
    const err = error as Error;
    log(`Warning: Could not download wheel: ${err.message}`, 'yellow');
    return false;
  }
}

function installDependencies(wheelName: string | false): boolean {
  log('\n=== Installing Dependencies ===', 'cyan');

  try {
    if (wheelName) {
      log(`Installing wheel: ${wheelName}`);
      execSyncSafe(`uv pip install ${wheelName}`);
      log('Wheel installed successfully.', 'green');

      log('Installing additional dependencies...');
      execSyncSafe('uv sync');
    } else {
      log('Installing all dependencies with uv sync...');
      execSyncSafe('uv sync --all-extras');
    }

    log('Dependencies installed successfully.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error installing dependencies: ${err.message}`, 'red');
    return false;
  }
}

function checkAdvancedBuild(): boolean {
  return (
    process.env.RLM_BUILD_FROM_SOURCE === '1' || process.argv.includes('--build-from-source')
  );
}

async function main(): Promise<void> {
  log('\n=========================================', 'cyan');
  log('  RLM-Claude-Code Setup', 'cyan');
  log('=========================================', 'cyan');

  const isAdvanced = checkAdvancedBuild();

  initGitSubmodules();

  if (!createVenv()) {
    process.exit(1);
  }

  if (isAdvanced) {
    log('\nAdvanced mode: Building from source...', 'cyan');
    log('Run: npm run build', 'yellow');
  } else {
    await downloadBinaries();
  }

  let wheelName: string | false = null;
  if (!isAdvanced) {
    wheelName = await downloadWheel();
  }

  if (!installDependencies(wheelName)) {
    process.exit(1);
  }

  log('\n=========================================', 'green');
  log('  Setup Complete!', 'green');
  log('=========================================', 'green');
  log('\nNext steps:');
  log('  1. Run: npm run verify    # Verify installation');
  log('  2. Run: npm run test      # Run test suite');
  log('  3. Restart Claude Code if using as plugin');
  log('');
}

main().catch((error) => {
  const err = error as Error;
  log(`\nSetup failed: ${err.message}`, 'red');
  process.exit(1);
});
