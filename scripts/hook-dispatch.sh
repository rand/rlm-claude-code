#!/bin/bash
# hook-dispatch.sh - Platform-aware hook dispatcher
#
# Usage: hook-dispatch.sh <hook-name>
#
# This script detects the current platform and executes the appropriate
# pre-built binary. Falls back to Python if binary not found.

set -e

# Determine plugin root
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
HOOK_NAME="$1"

if [ -z "$HOOK_NAME" ]; then
    echo "Usage: $0 <hook-name>" >&2
    exit 1
fi

# Check for legacy mode
if [ "${RLM_USE_LEGACY_HOOKS}" = "1" ]; then
    PYTHON_SCRIPT="${PLUGIN_ROOT}/scripts/legacy/${HOOK_NAME}.py"
    if [ -f "$PYTHON_SCRIPT" ]; then
        VENV_PYTHON="${PLUGIN_ROOT}/.venv/bin/python3"
        if [ -x "$VENV_PYTHON" ]; then
            exec "$VENV_PYTHON" "$PYTHON_SCRIPT"
        fi
        exec python3 "$PYTHON_SCRIPT"
    fi
fi

# Detect OS
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
    mingw*|msys*|cygwin*) OS="windows" ;;
esac

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    i386|i686) ARCH="386" ;;
esac

# Construct binary path
if [ "$OS" = "windows" ]; then
    BINARY="${PLUGIN_ROOT}/bin/${HOOK_NAME}-${OS}-${ARCH}.exe"
else
    BINARY="${PLUGIN_ROOT}/bin/${HOOK_NAME}-${OS}-${ARCH}"
fi

# Debug output
if [ "$HOOK_DEBUG" = "1" ]; then
    echo "[DEBUG] OS=$OS ARCH=$ARCH" >&2
    echo "[DEBUG] Looking for: $BINARY" >&2
fi

# Try binary first
if [ -x "$BINARY" ]; then
    exec "$BINARY"
fi

# Try platform-agnostic binary (dev build)
BINARY_DEV="${PLUGIN_ROOT}/bin/${HOOK_NAME}"
if [ -x "$BINARY_DEV" ]; then
    exec "$BINARY_DEV"
fi

# Fallback to Python
PYTHON_SCRIPT="${PLUGIN_ROOT}/scripts/${HOOK_NAME}.py"
if [ -f "$PYTHON_SCRIPT" ]; then
    if [ "$HOOK_DEBUG" = "1" ]; then
        echo "[DEBUG] Falling back to Python: $PYTHON_SCRIPT" >&2
    fi

    VENV_PYTHON="${PLUGIN_ROOT}/.venv/bin/python3"
    if [ -x "$VENV_PYTHON" ]; then
        exec "$VENV_PYTHON" "$PYTHON_SCRIPT"
    fi

    exec python3 "$PYTHON_SCRIPT"
fi

# Legacy script location
LEGACY_SCRIPT="${PLUGIN_ROOT}/scripts/legacy/${HOOK_NAME}.py"
if [ -f "$LEGACY_SCRIPT" ]; then
    exec python3 "$LEGACY_SCRIPT"
fi

# Nothing found
echo "Hook not found: $HOOK_NAME" >&2
echo "Searched:" >&2
echo "  - $BINARY" >&2
echo "  - $PYTHON_SCRIPT" >&2
exit 1
