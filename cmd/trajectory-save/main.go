// Command trajectory-save saves RLM trajectory data on session end.
package main

import (
	"time"

	"github.com/rand/rlm-claude-code/internal/events"
	"github.com/rand/rlm-claude-code/internal/hookio"
)

func main() {
	// Try to read input but don't fail if it's missing/invalid
	// Stop hooks may not always provide JSON input
	_, err := hookio.ReadInput()
	if err != nil {
		hookio.Debug("No valid input (expected for Stop hooks): %v", err)
		// Continue anyway - fail forward
	}

	hookio.Debug("Saving trajectory...")

	// Emit trajectory saved event
	events.Emit(map[string]any{
		"type":      "trajectory_saved",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"source":    "rlm-claude-code",
	}, "rlm-claude-code")

	// Output success for Claude Code
	hookio.Approve("")
}
