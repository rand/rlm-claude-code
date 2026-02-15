#!/usr/bin/env npx ts-node
/**
 * download-wheel.ts
 *
 * Download pre-built Python wheel from GitHub releases.
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { ROOT_DIR, log, getPlatform } from './common';

interface GitHubRelease {
  tag_name: string;
  assets: Array<{
    name: string;
    browser_download_url: string;
  }>;
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

function getWheelPlatform(os: string, cpu: string): string {
  if (os === 'darwin' && cpu === 'arm64') {
    return 'macosx_11_0_arm64';
  } else if (os === 'darwin' && cpu === 'amd64') {
    return 'macosx_10_9_x86_64';
  } else if (os === 'linux' && cpu === 'amd64') {
    return 'manylinux_2_17_x86_64';
  } else if (os === 'linux' && cpu === 'arm64') {
    return 'manylinux_2_17_aarch64';
  } else if (os === 'windows') {
    return 'win_amd64';
  }
  return 'unknown';
}

async function main(): Promise<void> {
  log('\n=========================================', 'cyan');
  log('  Download Python Wheel', 'cyan');
  log('=========================================', 'cyan');

  const { os, cpu } = getPlatform();
  const wheelPlatform = getWheelPlatform(os, cpu);

  log(`\nPlatform: ${wheelPlatform}`, 'dim');

  try {
    log('Fetching latest release information...', 'yellow');
    const release = await getLatestRelease();
    const version = release.tag_name;
    log(`Latest version: ${version}`, 'green');

    // Find matching wheel
    const wheelAsset = release.assets.find((a) => {
      return (
        a.name.includes('rlm_claude_code') &&
        a.name.endsWith('.whl') &&
        (a.name.includes(wheelPlatform) || a.name.includes('py3-none-any'))
      );
    });

    if (!wheelAsset) {
      log(`\nNo pre-built wheel found for ${wheelPlatform}`, 'yellow');
      log('Will need to build from source.', 'yellow');
      process.exit(1);
    }

    const wheelPath = path.join(ROOT_DIR, wheelAsset.name);
    log(`\nDownloading ${wheelAsset.name}...`, 'yellow');

    await downloadFile(wheelAsset.browser_download_url, wheelPath);

    log(`\n[OK] Downloaded: ${wheelAsset.name}`, 'green');
    log('Install with: uv pip install ' + wheelAsset.name, 'dim');

    log('\n=========================================', 'green');
    log('  Download Complete!', 'green');
    log('=========================================', 'green');
  } catch (error) {
    const err = error as Error;
    log(`\nDownload failed: ${err.message}`, 'red');
    process.exit(1);
  }
}

main();
