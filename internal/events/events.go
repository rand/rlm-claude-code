// Package events provides utilities for emitting and consuming cross-plugin events.
package events

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// Event is the base structure for all events.
type Event struct {
	Type          string    `json:"type"`
	Timestamp     time.Time `json:"timestamp"`
	Source        string    `json:"source"`
	SessionID     string    `json:"session_id,omitempty"`
	CorrelationID string    `json:"correlation_id,omitempty"`
}

// PhaseTransitionEvent is emitted when disciplined-process changes phase.
type PhaseTransitionEvent struct {
	Event
	FromPhase        string   `json:"from_phase"`
	ToPhase          string   `json:"to_phase"`
	TaskID           string   `json:"task_id,omitempty"`
	SpecRefs         []string `json:"spec_refs,omitempty"`
	ValidationPassed bool     `json:"validation_passed"`
}

// ModeChangeEvent is emitted when RLM mode changes.
type ModeChangeEvent struct {
	Event
	FromMode    string `json:"from_mode"`
	ToMode      string `json:"to_mode"`
	Reason      string `json:"reason"`
	TriggeredBy string `json:"triggered_by"` // "user", "auto", "dp_phase"
}

// --- Event Directory ---

func eventsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	dir := filepath.Join(home, ".claude", "events")
	os.MkdirAll(dir, 0755)
	return dir
}

// --- Emit ---

// Emit writes an event to both the log file and latest file.
func Emit(event any, source string) error {
	dir := eventsDir()

	// Append to log
	logFile := filepath.Join(dir, source+"-events.jsonl")
	f, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := json.NewEncoder(f).Encode(event); err != nil {
		return err
	}

	// Write latest
	latestFile := filepath.Join(dir, source+"-latest.json")
	data, err := json.MarshalIndent(event, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(latestFile, data, 0644)
}

// --- Consume ---

// ReadLatest reads the most recent event from a source.
func ReadLatest(source string) (map[string]any, error) {
	dir := eventsDir()
	latestFile := filepath.Join(dir, source+"-latest.json")

	data, err := os.ReadFile(latestFile)
	if err != nil {
		return nil, err
	}

	var event map[string]any
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	return event, nil
}

// GetDPPhase returns the current disciplined-process phase, or "unknown".
func GetDPPhase() string {
	event, err := ReadLatest("disciplined-process")
	if err != nil {
		return "unknown"
	}

	if event["type"] == "phase_transition" {
		if phase, ok := event["to_phase"].(string); ok {
			return phase
		}
	}
	return "unknown"
}

// GetRLMMode returns the current RLM mode, or "unknown".
func GetRLMMode() string {
	event, err := ReadLatest("rlm-claude-code")
	if err != nil {
		return "unknown"
	}

	if event["type"] == "mode_change" {
		if mode, ok := event["to_mode"].(string); ok {
			return mode
		}
	}
	return "unknown"
}

// --- Helper for DP → RLM coordination ---

// SuggestedRLMMode returns the suggested RLM mode based on DP phase.
func SuggestedRLMMode() string {
	phase := GetDPPhase()
	switch phase {
	case "spec", "review":
		return "thorough"
	case "test", "implement":
		return "balanced"
	case "orient", "decide":
		return "balanced"
	default:
		return "" // No suggestion
	}
}
