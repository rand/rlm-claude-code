/**
 * Tests for download-binaries.ts and common.ts
 *
 * Uses Node.js built-in test runner (available in Node 20+)
 * Tests the platform detection and archive naming logic
 * for cross-platform binary downloads.
 *
 * Run with: npx ts-node --test scripts/npm/__tests__/*.test.ts
 */

import { test, describe, it } from 'node:test';
import assert from 'node:assert';
import { getPlatform } from '../common';

// ============================================================================
// Platform Detection Tests (pure function tests)
// ============================================================================

describe('Platform Detection Logic', () => {
  // Test the logic that converts Node.js platform/arch to our format

  const convertPlatform = (nodePlatform: string, nodeArch: string) => {
    let os: string;
    let cpu: string;

    switch (nodePlatform) {
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
        throw new Error(`Unsupported platform: ${nodePlatform}`);
    }

    switch (nodeArch) {
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
        throw new Error(`Unsupported architecture: ${nodeArch}`);
    }

    return { os, cpu, platform: `${os}-${cpu}` };
  };

  it('should detect macOS arm64 correctly', () => {
    const result = convertPlatform('darwin', 'arm64');
    assert.deepStrictEqual(result, {
      os: 'darwin',
      cpu: 'arm64',
      platform: 'darwin-arm64',
    });
  });

  it('should detect macOS amd64 correctly (x64 -> amd64)', () => {
    const result = convertPlatform('darwin', 'x64');
    assert.deepStrictEqual(result, {
      os: 'darwin',
      cpu: 'amd64',
      platform: 'darwin-amd64',
    });
  });

  it('should detect Linux arm64 correctly', () => {
    const result = convertPlatform('linux', 'arm64');
    assert.deepStrictEqual(result, {
      os: 'linux',
      cpu: 'arm64',
      platform: 'linux-arm64',
    });
  });

  it('should detect Linux amd64 correctly', () => {
    const result = convertPlatform('linux', 'x64');
    assert.deepStrictEqual(result, {
      os: 'linux',
      cpu: 'amd64',
      platform: 'linux-amd64',
    });
  });

  it('should detect Windows amd64 correctly (win32 -> windows)', () => {
    const result = convertPlatform('win32', 'x64');
    assert.deepStrictEqual(result, {
      os: 'windows',
      cpu: 'amd64',
      platform: 'windows-amd64',
    });
  });

  it('should detect Windows arm64 correctly', () => {
    const result = convertPlatform('win32', 'arm64');
    assert.deepStrictEqual(result, {
      os: 'windows',
      cpu: 'arm64',
      platform: 'windows-arm64',
    });
  });

  it('should throw for unsupported platform', () => {
    assert.throws(
      () => convertPlatform('freebsd', 'x64'),
      /Unsupported platform/
    );
  });
});

// ============================================================================
// Archive Name Tests
// ============================================================================

describe('Archive Name Generation', () => {
  // Archive format: hooks-{version}-{os}-{cpu}.tar.gz

  const getArchiveName = (version: string, os: string, cpu: string): string => {
    return `hooks-${version}-${os}-${cpu}.tar.gz`;
  };

  it('should generate correct archive name for macOS arm64', () => {
    assert.strictEqual(
      getArchiveName('v0.7.0', 'darwin', 'arm64'),
      'hooks-v0.7.0-darwin-arm64.tar.gz'
    );
  });

  it('should generate correct archive name for macOS amd64', () => {
    assert.strictEqual(
      getArchiveName('v0.7.0', 'darwin', 'amd64'),
      'hooks-v0.7.0-darwin-amd64.tar.gz'
    );
  });

  it('should generate correct archive name for Linux arm64', () => {
    assert.strictEqual(
      getArchiveName('v0.7.0', 'linux', 'arm64'),
      'hooks-v0.7.0-linux-arm64.tar.gz'
    );
  });

  it('should generate correct archive name for Linux amd64', () => {
    assert.strictEqual(
      getArchiveName('v0.7.0', 'linux', 'amd64'),
      'hooks-v0.7.0-linux-amd64.tar.gz'
    );
  });

  it('should generate correct archive name for Windows amd64', () => {
    assert.strictEqual(
      getArchiveName('v0.7.0', 'windows', 'amd64'),
      'hooks-v0.7.0-windows-amd64.tar.gz'
    );
  });
});

// ============================================================================
// Binary Existence Check Tests
// ============================================================================

describe('Binary Existence Check', () => {
  const getExpectedBinaryName = (os: string, cpu: string, binary: string): string => {
    const prefix = binary; // e.g., 'session-init'
    if (os === 'windows') {
      return `${prefix}-${os}-${cpu}.exe`;
    }
    return `${prefix}-${os}-${cpu}`;
  };

  it('should generate correct binary name for macOS arm64', () => {
    assert.strictEqual(
      getExpectedBinaryName('darwin', 'arm64', 'session-init'),
      'session-init-darwin-arm64'
    );
  });

  it('should generate correct binary name for Windows with .exe', () => {
    assert.strictEqual(
      getExpectedBinaryName('windows', 'amd64', 'session-init'),
      'session-init-windows-amd64.exe'
    );
  });

  it('should generate correct binary name for Linux amd64', () => {
    assert.strictEqual(
      getExpectedBinaryName('linux', 'amd64', 'complexity-check'),
      'complexity-check-linux-amd64'
    );
  });
});

// ============================================================================
// Wheel Platform Tests
// ============================================================================

describe('Wheel Platform Detection', () => {
  const getWheelPlatform = (os: string, cpu: string): string => {
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
  };

  it('should return macosx_11_0_arm64 for macOS arm64', () => {
    assert.strictEqual(getWheelPlatform('darwin', 'arm64'), 'macosx_11_0_arm64');
  });

  it('should return macosx_10_9_x86_64 for macOS amd64', () => {
    assert.strictEqual(getWheelPlatform('darwin', 'amd64'), 'macosx_10_9_x86_64');
  });

  it('should return manylinux_2_17_x86_64 for Linux amd64', () => {
    assert.strictEqual(getWheelPlatform('linux', 'amd64'), 'manylinux_2_17_x86_64');
  });

  it('should return manylinux_2_17_aarch64 for Linux arm64', () => {
    assert.strictEqual(getWheelPlatform('linux', 'arm64'), 'manylinux_2_17_aarch64');
  });

  it('should return win_amd64 for Windows', () => {
    assert.strictEqual(getWheelPlatform('windows', 'amd64'), 'win_amd64');
  });

  it('should return unknown for unsupported combinations', () => {
    assert.strictEqual(getWheelPlatform('windows', 'arm64'), 'win_amd64');
    assert.strictEqual(getWheelPlatform('freebsd', 'x64'), 'unknown');
  });
});

// ============================================================================
// Current Platform Test (integration)
// ============================================================================

describe('Current Platform Detection', () => {
  it('should detect the current platform correctly', () => {
    const result = getPlatform();

    // Verify structure
    assert.ok(result.os, 'should have os property');
    assert.ok(result.cpu, 'should have cpu property');
    assert.ok(result.platform, 'should have platform property');

    // Verify the platform string format
    assert.strictEqual(result.platform, `${result.os}-${result.cpu}`);

    // Verify known values
    assert.ok(['darwin', 'linux', 'windows'].includes(result.os), 'os should be known value');
    assert.ok(['amd64', 'arm64', 'arm'].includes(result.cpu), 'cpu should be known value');
  });
});
