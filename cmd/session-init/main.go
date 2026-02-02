// Command session-init initializes the RLM environment on session start.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/rand/rlm-claude-code/internal/config"
	"github.com/rand/rlm-claude-code/internal/events"
	"github.com/rand/rlm-claude-code/internal/hookio"
)

func main() {
	input, err := hookio.ReadInput()
	if err != nil {
		hookio.Debug("Failed to read input: %v", err)
		os.Exit(1)
	}

	hookio.Debug("Session start: source=%s, cwd=%s", input.Source, input.CWD)

	pluginRoot := os.Getenv("CLAUDE_PLUGIN_ROOT")
	if pluginRoot == "" {
		hookio.SessionContext("RLM: plugin root not found")
		return
	}

	homeDir, _ := os.UserHomeDir()
	markerFile := filepath.Join(homeDir, ".claude", "plugins", ".rlm-initialized")
	venvPath := filepath.Join(pluginRoot, ".venv")

	// Fast path: already initialized
	if _, err := os.Stat(markerFile); err == nil {
		hookio.Debug("Already initialized, fast path")

		// Load config (handles migration)
		cfg, err := config.Load()
		if err != nil {
			hookio.Debug("Config load error: %v", err)
		} else {
			hookio.Debug("Config loaded: version=%s, mode=%s", cfg.Version, cfg.Activation.Mode)
		}

		writeEnvFile(venvPath)

		// Check DP phase for mode suggestion
		suggestion := getDPSuggestion()
		hookio.SessionContext("RLM initialized" + suggestion)
		return
	}

	// First run: bootstrap
	hookio.Debug("First run, bootstrapping...")

	// Ensure uv is available
	if !ensureUV() {
		hookio.SessionContext("RLM: uv installation failed, manual setup required")
		return
	}

	// Create venv if missing
	if _, err := os.Stat(venvPath); os.IsNotExist(err) {
		hookio.Debug("Creating venv at %s", venvPath)
		cmd := exec.Command("uv", "venv", venvPath, "--python", "3.12")
		cmd.Dir = pluginRoot
		if output, err := cmd.CombinedOutput(); err != nil {
			hookio.Debug("Failed to create venv: %v\n%s", err, output)
		}
	}

	// Install dependencies if requirements.txt exists
	reqFile := filepath.Join(pluginRoot, "requirements.txt")
	if _, err := os.Stat(reqFile); err == nil {
		hookio.Debug("Installing dependencies from %s", reqFile)
		cmd := exec.Command("uv", "pip", "install", "-r", reqFile)
		cmd.Env = append(os.Environ(), "VIRTUAL_ENV="+venvPath)
		cmd.Dir = pluginRoot
		if output, err := cmd.CombinedOutput(); err != nil {
			hookio.Debug("Failed to install dependencies: %v\n%s", err, output)
		}
	}

	// Verify rlm_core is importable (Rust backend)
	checkCmd := exec.Command(filepath.Join(venvPath, "bin", "python"), "-c", "import rlm_core")
	checkCmd.Dir = pluginRoot
	if output, err := checkCmd.CombinedOutput(); err != nil {
		hookio.Debug("rlm_core not available: %v\n%s (Python fallback will be used)", err, output)
	} else {
		hookio.Debug("rlm_core Rust backend verified")
	}

	// Create marker file
	os.MkdirAll(filepath.Dir(markerFile), 0755)
	os.WriteFile(markerFile, []byte(time.Now().Format(time.RFC3339)), 0644)

	// Load config (handles migration)
	cfg, err := config.Load()
	if err != nil {
		hookio.Debug("Config load error: %v", err)
	} else {
		hookio.Debug("Config loaded: version=%s, mode=%s", cfg.Version, cfg.Activation.Mode)
	}

	// Write to CLAUDE_ENV_FILE
	writeEnvFile(venvPath)

	// Emit initialization event
	events.Emit(map[string]any{
		"type":      "rlm_initialized",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"source":    "rlm-claude-code",
		"first_run": true,
	}, "rlm-claude-code")

	hookio.SessionContext("RLM initialized (first run)")
}

func ensureUV() bool {
	if _, err := exec.LookPath("uv"); err == nil {
		return true
	}

	hookio.Debug("uv not found, attempting installation...")

	cmd := exec.Command("sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh")
	if err := cmd.Run(); err != nil {
		hookio.Debug("Failed to install uv: %v", err)
		return false
	}

	if _, err := exec.LookPath("uv"); err != nil {
		paths := []string{
			filepath.Join(os.Getenv("HOME"), ".local", "bin", "uv"),
			filepath.Join(os.Getenv("HOME"), ".cargo", "bin", "uv"),
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				os.Setenv("PATH", filepath.Dir(p)+":"+os.Getenv("PATH"))
				return true
			}
		}
		return false
	}

	return true
}

func writeEnvFile(venvPath string) {
	envFile := os.Getenv("CLAUDE_ENV_FILE")
	if envFile == "" {
		hookio.Debug("CLAUDE_ENV_FILE not set, skipping venv activation")
		return
	}

	f, err := os.OpenFile(envFile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		hookio.Debug("Failed to open CLAUDE_ENV_FILE: %v", err)
		return
	}
	defer f.Close()

	activatePath := filepath.Join(venvPath, "bin", "activate")
	fmt.Fprintf(f, "source '%s'\n", activatePath)
	hookio.Debug("Wrote venv activation to %s", envFile)
}

func getDPSuggestion() string {
	mode := events.SuggestedRLMMode()
	if mode == "" {
		return ""
	}

	phase := events.GetDPPhase()
	if phase == "unknown" {
		return ""
	}

	return fmt.Sprintf(" (DP phase: %s → consider /rlm mode %s)", phase, mode)
}
