// Command trajectory-save saves RLM trajectory data on session end.
package main

import (
	"os"
	"time"

	"github.com/rand/rlm-claude-code/internal/events"
	"github.com/rand/rlm-claude-code/internal/hookio"
)

func main() {
	_, err := hookio.ReadInput()
	if err != nil {
		hookio.Debug("Failed to read input: %v", err)
		os.Exit(1)
	}

	hookio.Debug("Saving trajectory...")

	// Emit trajectory saved event
	events.Emit(map[string]any{
		"type":      "trajectory_saved",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"source":    "rlm-claude-code",
	}, "rlm-claude-code")
}
