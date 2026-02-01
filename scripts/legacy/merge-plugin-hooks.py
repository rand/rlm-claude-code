#!/usr/bin/env python3
"""
Merge Plugin Hooks into Claude Code Settings

This script reads hooks from enabled plugins and merges them into settings.json.
Run this after installing or updating plugins to ensure their hooks are active.

Design:
- settings.user.json: Your manual settings (edit this file)
- settings.json: Generated file (don't edit directly)
- Plugin hooks are read from each plugin's hooks/hooks.json

Usage:
    python3 ~/.claude/scripts/merge-plugin-hooks.py [--dry-run]

================================================================================
WORKAROUND: `uv run` incompatible with macOS sandbox (as of Jan 2025)
================================================================================

PROBLEM:
    `uv run python ...` crashes inside Claude Code's macOS sandbox with:

        thread 'main2' panicked at system-configuration-0.6.1/src/dynamic_store.rs:154:1:
        Attempted to create a NULL object.

    This happens because:
    1. `uv` uses the `hyper-util` crate for HTTP
    2. `hyper-util` uses `system-configuration-rs` for macOS proxy detection
    3. `system-configuration-rs` calls macOS SCDynamicStore APIs
    4. The sandbox blocks these system API calls, causing a NULL return
    5. The crate panics on NULL (fixed in git but not released)

ROOT CAUSE CHAIN:
    uv → reqwest → hyper-rustls → hyper-util → system-configuration-rs
                                                      ↓
                                            SCDynamicStoreCreate()
                                                      ↓
                                            Returns NULL in sandbox
                                                      ↓
                                            Rust panic!

FIX STATUS (check these for updates):
    1. system-configuration-rs: Fix merged (commit de5966a, Sep 2024)
       BUT released as 0.7.0 with API changes
       Repo: https://github.com/mullvad/system-configuration-rs

    2. hyper-util: NOT YET UPDATED to handle new API
       Repo: https://github.com/hyperium/hyper-util
       Need to check for release > 0.1.19 with system-configuration 0.7.0 support

    3. uv: Tracking issue https://github.com/astral-sh/uv/issues/16916
       Blocked on hyper-util update

WORKAROUND IMPLEMENTED HERE:
    This script replaces `uv run python` with direct `.venv/bin/python` calls.
    The .venv already exists (created by uv outside sandbox) and works fine.

TO CHECK IF FIX IS AVAILABLE:
    1. Try: `uv run python -c "print('test')"` inside a Claude Code session
    2. If it works without panic, the fix has landed
    3. Remove UV_RUN_WORKAROUND_ENABLED below and re-run this script

TO REVERT THIS WORKAROUND:
    Set UV_RUN_WORKAROUND_ENABLED = False below, then re-run:
        python3 ~/.claude/scripts/merge-plugin-hooks.py

================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


# =============================================================================
# WORKAROUND TOGGLE - Set to False when uv sandbox issue is fixed
# =============================================================================
UV_RUN_WORKAROUND_ENABLED = True
# =============================================================================


CLAUDE_DIR = Path.home() / ".claude"
SETTINGS_USER = CLAUDE_DIR / "settings.user.json"
SETTINGS_OUTPUT = CLAUDE_DIR / "settings.json"
INSTALLED_PLUGINS = CLAUDE_DIR / "plugins" / "installed_plugins.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file, return empty dict if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save JSON with pretty formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def get_enabled_plugins(settings: dict[str, Any]) -> set[str]:
    """Get set of enabled plugin identifiers."""
    enabled = settings.get("enabledPlugins", {})
    return {k for k, v in enabled.items() if v}


def normalize_plugin_hooks(hooks_data: Any) -> dict[str, list]:
    """
    Normalize plugin hooks to standard format: {event: [entries]}.

    Handles two formats:
    1. Dict format: {"SessionStart": [...], "PreToolUse": [...]}
    2. List format: [{"event": "PreToolUse", "matcher": "...", "hooks": [...]}]
    """
    if isinstance(hooks_data, dict):
        # Already in dict format
        normalized: dict[str, list] = {}
        for event, entries in hooks_data.items():
            if isinstance(entries, list):
                normalized[event] = entries
            else:
                normalized[event] = [entries]
        return normalized

    elif isinstance(hooks_data, list):
        # List format - convert to dict
        normalized = {}
        for entry in hooks_data:
            event = entry.get("event")
            if not event:
                continue
            if event not in normalized:
                normalized[event] = []
            # The entry itself becomes the hook config
            normalized[event].append(entry)
        return normalized

    return {}


def find_plugin_hooks(plugin_id: str, installed: dict[str, Any]) -> dict[str, Any] | None:
    """Find and load hooks.json for a plugin."""
    plugins = installed.get("plugins", {})

    if plugin_id not in plugins:
        return None

    installs = plugins[plugin_id]
    if not installs:
        return None

    # Get the most recent installation
    install = installs[0]
    install_path = Path(install.get("installPath", ""))

    if not install_path.exists():
        return None

    # Look for hooks.json in standard locations
    hooks_paths = [
        install_path / "hooks" / "hooks.json",
        install_path / "hooks.json",
    ]

    for hooks_path in hooks_paths:
        if hooks_path.exists():
            try:
                with open(hooks_path) as f:
                    data = json.load(f)
                    # Get hooks data and normalize to dict format
                    raw_hooks = data.get("hooks", data)
                    normalized = normalize_plugin_hooks(raw_hooks)
                    return {
                        "path": str(hooks_path),
                        "hooks": normalized,
                        "plugin": plugin_id,
                        "install_path": str(install_path),
                    }
            except json.JSONDecodeError:
                print(f"  Warning: Invalid JSON in {hooks_path}", file=sys.stderr)

    return None


def fix_uv_run_for_sandbox(cmd: str, install_path: str) -> str:
    """
    Replace `uv run python` with direct .venv/bin/python for sandbox compatibility.

    WORKAROUND for: https://github.com/astral-sh/uv/issues/16916

    `uv run` crashes in macOS sandbox due to system-configuration-rs calling
    SCDynamicStore APIs that return NULL when sandboxed.

    This function:
    1. Detects `uv run python` patterns in hook commands
    2. Replaces with direct .venv/bin/python path (which works in sandbox)
    3. Preserves the rest of the command unchanged

    Args:
        cmd: The original hook command
        install_path: The plugin's installation directory (contains .venv)

    Returns:
        Modified command with uv run replaced, or original if no match
    """
    if not UV_RUN_WORKAROUND_ENABLED:
        return cmd

    venv_python = Path(install_path) / ".venv" / "bin" / "python"

    # Check if .venv exists before applying workaround
    if not venv_python.exists():
        # No .venv available, can't apply workaround
        # This might still fail in sandbox, but we can't fix it here
        return cmd

    # Pattern: `uv run python "path/to/script.py"` or `uv run python script.py`
    # Also handles: `cd "..." && uv run python ...`

    # Replace `uv run python` with the venv python path
    # Be careful to handle quoted paths
    patterns = [
        # uv run python "quoted/path"
        (r'uv run python\s+"([^"]+)"', f'"{venv_python}" "\\1"'),
        # uv run python unquoted/path (capture until space or end)
        (r'uv run python\s+(\S+)', f'"{venv_python}" \\1'),
    ]

    modified = cmd
    for pattern, replacement in patterns:
        modified = re.sub(pattern, replacement, modified)

    # If we made a change, add a marker comment for debugging
    if modified != cmd:
        # Log the transformation (visible in verbose mode)
        pass  # Could add logging here if needed

    return modified


def normalize_hook_entry(entry: dict[str, Any], plugin_id: str, install_path: str) -> dict[str, Any]:
    """Normalize a hook entry and add source marker."""
    normalized = deepcopy(entry)

    # Add source marker (Claude Code ignores unknown fields)
    normalized["_source"] = f"plugin:{plugin_id}"

    # Fix relative paths in hook commands
    # Replace ${CLAUDE_PLUGIN_ROOT} with actual install path
    if "hooks" in normalized:
        for hook in normalized["hooks"]:
            if "command" in hook:
                cmd = hook["command"]
                # Replace plugin root variable
                cmd = cmd.replace("${CLAUDE_PLUGIN_ROOT}", install_path)
                cmd = cmd.replace("$CLAUDE_PLUGIN_ROOT", install_path)
                # Handle relative script paths (e.g., "scripts/foo.sh")
                if not cmd.startswith("/") and not cmd.startswith("cd "):
                    # Check if it looks like a relative path to a script
                    first_word = cmd.split()[0] if cmd.split() else ""
                    if "/" in first_word and not first_word.startswith("$"):
                        cmd = f'cd "{install_path}" && {cmd}'

                # Apply uv run workaround for sandbox compatibility
                cmd = fix_uv_run_for_sandbox(cmd, install_path)

                hook["command"] = cmd

    return normalized


def merge_hooks(
    user_hooks: dict[str, list],
    plugin_hooks_list: list[dict[str, Any]],
) -> dict[str, list]:
    """Merge user hooks with plugin hooks."""
    merged: dict[str, list] = {}

    # Start with user hooks (these take priority for ordering)
    for event, entries in user_hooks.items():
        merged[event] = []
        for entry in entries:
            # Mark user entries
            entry_copy = deepcopy(entry)
            if "_source" not in entry_copy:
                entry_copy["_source"] = "user"
            merged[event].append(entry_copy)

    # Add plugin hooks
    for plugin_data in plugin_hooks_list:
        plugin_id = plugin_data["plugin"]
        hooks = plugin_data["hooks"]
        install_path = plugin_data.get("install_path", "")

        for event, entries in hooks.items():
            if event not in merged:
                merged[event] = []

            # Handle both list and dict formats
            if isinstance(entries, list):
                for entry in entries:
                    merged[event].append(normalize_hook_entry(entry, plugin_id, install_path))
            elif isinstance(entries, dict):
                # Some plugins use dict format directly
                merged[event].append(normalize_hook_entry(entries, plugin_id, install_path))

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge plugin hooks into settings.json")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check if settings.user.json exists, create from settings.json if not
    if not SETTINGS_USER.exists():
        if SETTINGS_OUTPUT.exists():
            print(f"Creating {SETTINGS_USER} from existing {SETTINGS_OUTPUT}")
            # Copy current settings as the user base
            settings = load_json(SETTINGS_OUTPUT)
            # Remove any plugin-sourced hooks before saving as user settings
            if "hooks" in settings:
                clean_hooks: dict[str, list] = {}
                for event, entries in settings["hooks"].items():
                    clean_entries = [e for e in entries if e.get("_source", "user") == "user"]
                    if clean_entries:
                        # Remove _source marker from user entries for clean base
                        for entry in clean_entries:
                            entry.pop("_source", None)
                        clean_hooks[event] = clean_entries
                settings["hooks"] = clean_hooks
            save_json(SETTINGS_USER, settings)
        else:
            print(f"Error: Neither {SETTINGS_USER} nor {SETTINGS_OUTPUT} exists", file=sys.stderr)
            return 1

    # Load user settings
    user_settings = load_json(SETTINGS_USER)
    user_hooks = user_settings.get("hooks", {})

    # Load installed plugins info
    installed = load_json(INSTALLED_PLUGINS)

    # Get enabled plugins
    enabled = get_enabled_plugins(user_settings)

    if args.verbose:
        print(f"Enabled plugins: {', '.join(sorted(enabled))}")
        if UV_RUN_WORKAROUND_ENABLED:
            print("  [uv run workaround ENABLED - see script header for details]")

    # Find hooks from each enabled plugin
    plugin_hooks_list: list[dict[str, Any]] = []
    for plugin_id in sorted(enabled):
        hooks_data = find_plugin_hooks(plugin_id, installed)
        if hooks_data:
            plugin_hooks_list.append(hooks_data)
            if args.verbose:
                print(f"  Found hooks for {plugin_id}: {hooks_data['path']}")
                for event in hooks_data["hooks"]:
                    print(f"    - {event}")

    # Merge hooks
    merged_hooks = merge_hooks(user_hooks, plugin_hooks_list)

    # Build final settings
    final_settings = deepcopy(user_settings)
    final_settings["hooks"] = merged_hooks

    # Add generation marker
    final_settings["_generated"] = {
        "by": "merge-plugin-hooks.py",
        "user_file": str(SETTINGS_USER),
        "plugins_merged": [p["plugin"] for p in plugin_hooks_list],
        "uv_run_workaround": UV_RUN_WORKAROUND_ENABLED,
    }

    if args.dry_run:
        print("\n=== DRY RUN - Would write to settings.json ===")
        print(json.dumps(final_settings, indent=2))
        return 0

    # Write final settings
    save_json(SETTINGS_OUTPUT, final_settings)

    # Summary
    print(f"Merged hooks from {len(plugin_hooks_list)} plugins into {SETTINGS_OUTPUT}")
    for plugin_data in plugin_hooks_list:
        events = list(plugin_data["hooks"].keys())
        print(f"  - {plugin_data['plugin']}: {', '.join(events)}")

    if UV_RUN_WORKAROUND_ENABLED:
        print("\n[uv run → .venv/bin/python workaround applied]")
        print("  See script header for details on this workaround.")
        print("  To check if fix is available: uv run python -c \"print('test')\"")

    print(f"\nTo update your settings, edit {SETTINGS_USER} then re-run this script.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
