#!/usr/bin/env npx ts-node
/**
 * common.ts
 *
 * Shared utilities for RLM-Claude-Code npm scripts.
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

/**
 * Find the project root directory by looking for pyproject.toml
 * This is more reliable than __dirname when running via npx/ts-node
 */
function findRootDir(): string {
  // First, check for CLAUDE_PLUGIN_ROOT environment variable
  if (process.env.CLAUDE_PLUGIN_ROOT) {
    return process.env.CLAUDE_PLUGIN_ROOT;
  }

  // Start from current working directory and search up
  let dir = process.cwd();
  while (dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, 'pyproject.toml')) &&
        fs.existsSync(path.join(dir, 'package.json'))) {
      return dir;
    }
    dir = path.dirname(dir);
  }

  // Fallback to __dirname-based resolution
  return path.resolve(__dirname, '..', '..', '..');
}

export const ROOT_DIR = findRootDir();

export type ColorStyle = 'green' | 'red' | 'yellow' | 'cyan' | 'bold' | 'dim';

const ANSI_CODES: Record<ColorStyle, string> = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
};

const RESET = '\x1b[0m';

export function log(message: string, style: ColorStyle = 'dim'): void {
  const code = ANSI_CODES[style];
  console.log(`${code}${message}${RESET}`);
}

export interface RunCommandOptions {
  silent?: boolean;
  allowFailure?: boolean;
  cwd?: string;
}

export function runCommand(command: string, options: RunCommandOptions = {}): string | null {
  try {
    return execSync(command, {
      cwd: options.cwd ?? ROOT_DIR,
      encoding: 'utf8',
      stdio: options.silent ? 'pipe' : 'inherit',
    });
  } catch (error) {
    if (!options.allowFailure) {
      throw error;
    }
    return null;
  }
}

export function fileExists(filePath: string): boolean {
  return fs.existsSync(path.join(ROOT_DIR, filePath));
}

export function getPlatform(): { os: string; cpu: string; platform: string } {
  const platform = process.platform;
  const arch = process.arch;

  let os: string;
  let cpu: string;

  switch (platform) {
    case 'darwin':
      os = 'darwin';
      break;
    case 'linux':
      os = 'linux';
      break;
    case 'win32':
      os = 'windows';
      break;
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }

  switch (arch) {
    case 'x64':
      cpu = 'amd64';
      break;
    case 'arm64':
      cpu = 'arm64';
      break;
    case 'arm':
      cpu = 'arm';
      break;
    default:
      throw new Error(`Unsupported architecture: ${arch}`);
  }

  return { os, cpu, platform: `${os}-${cpu}` };
}

export interface CommandCheckResult {
  success: boolean;
  version?: string;
  error?: string;
  command?: string;
}

export function checkCommand(command: string, versionFlag = '--version'): CommandCheckResult {
  try {
    const result = execSync(`${command} ${versionFlag}`, {
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return { success: true, version: result.trim() };
  } catch (error) {
    const err = error as Error;
    return { success: false, error: err.message };
  }
}
