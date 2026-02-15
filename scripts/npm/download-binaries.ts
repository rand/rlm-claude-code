#!/usr/bin/env npx ts-node
/**
 * download-binaries.ts
 *
 * Download pre-built Go hook binaries from GitHub releases.
 * Handles tar.gz archives and extracts them to bin/ directory.
 * Cross-platform: works on Windows, macOS, Linux.
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import zlib from 'zlib';
import { pipeline } from 'stream';
import { promisify } from 'util';
import * as tar from 'tar';
import { execSync } from 'child_process';
import { ROOT_DIR, log, getPlatform } from './common';

const pipelineAsync = promisify(pipeline);

interface GitHubRelease {
  tag_name: string;
  assets: Array<{
    name: string;
    browser_download_url: string;
  }>;
}

/**
 * Get the archive name for the current platform
 */
function getArchiveName(version: string, os: string, cpu: string): string {
  // Release archives are named: hooks-v0.7.0-darwin-arm64.tar.gz
  return `hooks-${version}-${os}-${cpu}.tar.gz`;
}

/**
 * Download file from URL to destination
 */
function downloadFile(url: string, destPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    const file = fs.createWriteStream(destPath);

    const doDownload = (downloadUrl: string) => {
      protocol.get(downloadUrl, (response) => {
        if (response.statusCode === 301 || response.statusCode === 302) {
          const redirectUrl = response.headers.location;
          if (redirectUrl) {
            doDownload(redirectUrl);
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
    };

    doDownload(url);
  });
}

/**
 * Get latest release from GitHub
 */
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

/**
 * Remove Apple quarantine attributes on macOS
 * This is needed for binaries downloaded from the internet
 */
function removeAppleQuarantine(binDir: string): void {
  if (process.platform !== 'darwin') {
    return;
  }

  const files = fs.readdirSync(binDir).filter(f =>
    !f.endsWith('.tar.gz') && !f.endsWith('.txt')
  );

  let removed = 0;
  for (const file of files) {
    const filePath = path.join(binDir, file);
    try {
      // Remove com.apple.provenance
      execSync(`xattr -d com.apple.provenance "${filePath}" 2>/dev/null`, {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      removed++;
    } catch {
      // Attribute may not exist
    }

    try {
      // Remove com.apple.quarantine
      execSync(`xattr -d com.apple.quarantine "${filePath}" 2>/dev/null`, {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      removed++;
    } catch {
      // Attribute may not exist
    }
  }

  if (removed > 0) {
    log(`  Removed ${removed} Apple quarantine attributes`, 'dim');
  }
}

/**
 * Set executable permissions on Unix-like systems
 */
function setExecutablePermissions(binDir: string): void {
  if (process.platform === 'win32') {
    return;
  }

  const binaries = fs.readdirSync(binDir).filter(f =>
    !f.endsWith('.tar.gz') && !f.endsWith('.txt')
  );

  for (const binary of binaries) {
    const binaryPath = path.join(binDir, binary);
    try {
      fs.chmodSync(binaryPath, 0o755);
    } catch {
      // Ignore chmod errors
    }
  }
}

/**
 * Check if binaries already exist for the current platform
 */
function binariesExist(binDir: string, os: string, cpu: string): boolean {
  const expectedBinary = os === 'windows'
    ? `session-init-${os}-${cpu}.exe`
    : `session-init-${os}-${cpu}`;

  return fs.existsSync(path.join(binDir, expectedBinary));
}

/**
 * Extract tar.gz archive to bin directory
 */
async function extractArchive(archivePath: string, binDir: string): Promise<void> {
  await tar.x({
    file: archivePath,
    cwd: binDir,
    strip: 0, // Keep directory structure from archive
  });
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const force = args.includes('--force');

  log('\n=========================================', 'cyan');
  log('  Download Pre-built Binaries', 'cyan');
  log('=========================================', 'cyan');

  const { os, cpu, platform } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');

  log(`\nPlatform: ${platform}`, 'dim');

  // Create bin directory if needed
  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }

  // Check if already downloaded
  if (!force && binariesExist(binDir, os, cpu)) {
    log(`\nBinaries already exist for ${platform}.`, 'green');
    log('Use --force to re-download.', 'dim');
    return;
  }

  try {
    log('\nFetching latest release information...', 'yellow');
    const release = await getLatestRelease();
    const version = release.tag_name;
    log(`Latest version: ${version}`, 'green');

    // Find the archive for this platform
    const archiveName = getArchiveName(version, os, cpu);
    const asset = release.assets.find((a) => a.name === archiveName);

    if (!asset) {
      log(`\n[ERROR] Archive not found: ${archiveName}`, 'red');
      log(`Available platforms in release:`, 'yellow');
      release.assets
        .filter(a => a.name.endsWith('.tar.gz'))
        .forEach(a => log(`  - ${a.name}`, 'dim'));
      log('\nYou may need to build from source:', 'yellow');
      log('  npm run build -- --binaries-only', 'cyan');
      process.exit(1);
    }

    // Download the archive
    const archivePath = path.join(binDir, archiveName);
    log(`\nDownloading ${archiveName}...`, 'yellow');
    await downloadFile(asset.browser_download_url, archivePath);
    log(`  [OK] Downloaded`, 'green');

    // Extract
    log('\nExtracting binaries...', 'yellow');
    await extractArchive(archivePath, binDir);
    log('  [OK] Extracted', 'green');

    // Clean up archive
    fs.unlinkSync(archivePath);
    log('  [OK] Cleaned up archive', 'dim');

    // Set permissions
    setExecutablePermissions(binDir);
    log('  [OK] Set executable permissions', 'dim');

    // Remove Apple quarantine on macOS
    removeAppleQuarantine(binDir);

    // List extracted binaries
    log('\nBinaries installed:', 'green');
    const binaries = fs.readdirSync(binDir).filter(f =>
      f.startsWith('session-init') ||
      f.startsWith('complexity-check') ||
      f.startsWith('trajectory-save')
    );
    for (const binary of binaries) {
      log(`  - ${binary}`, 'dim');
    }

    log('\n=========================================', 'green');
    log('  Download Complete!', 'green');
    log('=========================================', 'green');
  } catch (error) {
    const err = error as Error;
    log(`\nDownload failed: ${err.message}`, 'red');
    log('\nYou may need to build from source:', 'yellow');
    log('  npm run build -- --binaries-only', 'cyan');
    process.exit(1);
  }
}

main();
