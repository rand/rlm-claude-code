// Command trajectory-save saves RLM trajectory data on session end.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/rand/rlm-claude-code/internal/events"
	"github.com/rand/rlm-claude-code/internal/hookio"
)

// saveTrajectory calls Python to persist the RLM session state and trajectory.
func saveTrajectory(pluginRoot string, venvPath string) {
	pythonPath := filepath.Join(venvPath, "bin", "python")

	// Check if python exists
	if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
		hookio.Debug("Python not found at %s, skipping trajectory save", pythonPath)
		return
	}

	script := fmt.Sprintf(`
import sys
sys.path.insert(0, '%s')
try:
    from src.state_persistence import get_persistence
    persistence = get_persistence()
    if persistence.current_state is not None:
        state_file = persistence.save_state()
        print(f"State saved to: {{state_file}}")
    else:
        print("No active session state to save")
except Exception as e:
    print(f"Trajectory save failed: {{e}}", file=sys.stderr)
`, pluginRoot)

	cmd := exec.Command(pythonPath, "-c", script)
	cmd.Dir = pluginRoot
	if output, err := cmd.CombinedOutput(); err != nil {
		hookio.Debug("Failed to save trajectory: %v\n%s", err, output)
	} else {
		hookio.Debug("Trajectory saved: %s", string(output))
	}
}

func main() {
	// Try to read input but don't fail if it's missing/invalid
	// Stop hooks may not always provide JSON input
	_, err := hookio.ReadInput()
	if err != nil {
		hookio.Debug("No valid input (expected for Stop hooks): %v", err)
		// Continue anyway - fail forward
	}

	hookio.Debug("Saving trajectory...")

	// Get plugin root and venv path
	pluginRoot := os.Getenv("CLAUDE_PLUGIN_ROOT")
	if pluginRoot != "" {
		venvPath := filepath.Join(pluginRoot, ".venv")
		saveTrajectory(pluginRoot, venvPath)
	}

	// Emit trajectory saved event
	events.Emit(map[string]any{
		"type":      "trajectory_saved",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"source":    "rlm-claude-code",
	}, "rlm-claude-code")

	// Output success for Claude Code
	hookio.Approve("")
}
