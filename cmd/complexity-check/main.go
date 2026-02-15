// Command complexity-check evaluates whether RLM should activate based on prompt complexity.
// It respects the activation mode from config:
//   - "always": Activate on all non-trivial prompts
//   - "never" / "manual": Never auto-activate
//   - "complexity" / "auto": Use complexity-based activation (default)
package main

import (
	"fmt"
	"os"

	"github.com/rand/rlm-claude-code/internal/classify"
	"github.com/rand/rlm-claude-code/internal/config"
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

	// Load config to check activation mode
	cfg, err := config.Load()
	if err != nil {
		hookio.Debug("Failed to load config: %v, using defaults", err)
	}

	// Determine activation mode (default to "auto" if config unavailable)
	activationMode := "auto"
	if cfg != nil {
		activationMode = cfg.Activation.Mode
	}
	hookio.Debug("Activation mode: %s", activationMode)

	// Handle mode-based activation
	switch activationMode {
	case "never", "manual":
		// Never auto-activate in manual/never mode
		hookio.Debug("Mode is %s: skipping activation", activationMode)
		hookio.Approve("")
		return

	case "always":
		// Activate on all non-trivial prompts
		if classify.IsFastPath(prompt) {
			hookio.Debug("Fast-path bypass (always mode)")
			hookio.Approve("")
			return
		}
		// Skip if trivial rigor
		rigor := events.GetDPRigor()
		if rigor == "trivial" {
			hookio.Approve("")
			return
		}
		// Always activate for non-trivial prompts
		dpPhase := events.GetDPPhase()
		mode := classify.SuggestMode(true, dpPhase)
		msg := fmt.Sprintf("[RLM %s mode: always-on]", mode)
		hookio.Debug("Activating: %s", msg)
		hookio.Approve(msg)
		events.Emit(map[string]any{
			"type":     "rlm_activation_suggested",
			"source":   "rlm-claude-code",
			"dp_phase": dpPhase,
			"reason":   "always_mode",
			"mode":     mode,
		}, "rlm-claude-code")
		return

	case "complexity", "auto":
		// Fall through to complexity-based logic below
	}

	// Complexity-based activation (default behavior)
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
