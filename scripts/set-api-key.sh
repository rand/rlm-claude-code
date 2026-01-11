#!/bin/bash
# Securely set the Anthropic API key
#
# Usage: ./scripts/set-api-key.sh [API_KEY]
#
# If no key provided, prompts for input (hidden)

set -e

ENV_FILE="$(dirname "$0")/../.env"

if [ -n "$1" ]; then
    API_KEY="$1"
else
    echo "Enter your Anthropic API key (input hidden):"
    read -s API_KEY
    echo
fi

if [ -z "$API_KEY" ]; then
    echo "Error: No API key provided"
    exit 1
fi

# Validate key format (basic check)
if [[ ! "$API_KEY" =~ ^sk-ant- ]]; then
    echo "Warning: Key doesn't start with 'sk-ant-', are you sure it's correct?"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Write to .env file
echo "ANTHROPIC_API_KEY=$API_KEY" > "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "API key saved to .env (file permissions: 600)"
echo "The key will be automatically loaded when running the orchestrator."
