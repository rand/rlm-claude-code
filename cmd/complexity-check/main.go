// Command complexity-check evaluates whether RLM should activate based on context complexity.
package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/rand/rlm-claude-code/internal/events"
	"github.com/rand/rlm-claude-code/internal/hookio"
)

func main() {
	input, err := hookio.ReadInput()
	if err != nil {
		hookio.Debug("Failed to read input: %v", err)
		os.Exit(1)
	}

	hookio.Debug("Complexity check: tool=%s", input.ToolName)

	threshold := 80000
	if env := os.Getenv("RLM_TOKEN_THRESHOLD"); env != "" {
		if t, err := strconv.Atoi(env); err == nil {
			threshold = t
		}
	}

	if os.Getenv("RLM_DISABLED") == "1" {
		hookio.Approve("")
		return
	}

	dpPhase := events.GetDPPhase()
	hookio.Debug("DP phase: %s", dpPhase)

	shouldActivate := false
	reason := ""

	suggestedMode := events.SuggestedRLMMode()
	if suggestedMode != "" {
		shouldActivate = true
		reason = fmt.Sprintf("DP phase %s: RLM %s mode recommended", dpPhase, suggestedMode)
	}

	if shouldActivate {
		hookio.Approve(reason)
		events.Emit(map[string]any{
			"type":      "rlm_activation_suggested",
			"source":    "rlm-claude-code",
			"dp_phase":  dpPhase,
			"reason":    reason,
			"threshold": threshold,
		}, "rlm-claude-code")
	} else {
		hookio.Approve("")
	}
}
