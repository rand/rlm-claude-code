// Command complexity-check evaluates whether RLM should activate based on prompt complexity.
package main

import (
	"fmt"
	"os"

	"github.com/rand/rlm-claude-code/internal/classify"
	"github.com/rand/rlm-claude-code/internal/events"
	"github.com/rand/rlm-claude-code/internal/hookio"
)

func main() {
	input, err := hookio.ReadInput()
	if err != nil {
		hookio.Debug("Failed to read input: %v", err)
		// Fail forward - approve and continue rather than blocking
		hookio.Approve("")
		return
	}

	if os.Getenv("RLM_DISABLED") == "1" {
		hookio.Approve("")
		return
	}

	prompt := input.UserPrompt
	hookio.Debug("Complexity check: prompt length=%d", len(prompt))

	// Skip if trivial rigor
	rigor := events.GetDPRigor()
	if rigor == "trivial" {
		hookio.Approve("")
		return
	}

	// Fast-path bypass for trivial prompts
	if classify.IsFastPath(prompt) {
		hookio.Debug("Fast-path bypass")
		hookio.Approve("")
		return
	}

	dpPhase := events.GetDPPhase()
	hookio.Debug("DP phase: %s, rigor: %s", dpPhase, rigor)

	activate, reason, mode := classify.ShouldActivate(prompt, dpPhase, rigor)

	if activate {
		msg := fmt.Sprintf("[RLM %s mode: %s]", mode, reason)
		hookio.Debug("Activating: %s", msg)
		hookio.Approve(msg)
		events.Emit(map[string]any{
			"type":     "rlm_activation_suggested",
			"source":   "rlm-claude-code",
			"dp_phase": dpPhase,
			"reason":   reason,
			"mode":     mode,
		}, "rlm-claude-code")
	} else {
		hookio.Debug("No activation: %s", reason)
		hookio.Approve("")
	}
}
