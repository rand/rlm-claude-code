# Makefile for Claude Code plugin hooks
# Usage:
#   make dev    - Build for current platform (fast)
#   make all    - Build for all platforms
#   make test   - Run tests
#   make clean  - Remove built binaries

GO := go
GOFLAGS := -ldflags="-s -w" -trimpath

# Commands to build (add more as you create them)
CMDS := session-init complexity-check trajectory-save

# Platforms to build for
PLATFORMS := darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64

.PHONY: all dev clean test lint help

# Default: build for current platform
dev:
	@mkdir -p bin
	@for cmd in $(CMDS); do \
		if [ -d "cmd/$$cmd" ]; then \
			echo "Building $$cmd..."; \
			$(GO) build $(GOFLAGS) -o bin/$$cmd ./cmd/$$cmd; \
		fi; \
	done
	@echo "Built binaries in bin/"
	@ls -la bin/

# Build for all platforms
all:
	@mkdir -p bin
	@for cmd in $(CMDS); do \
		if [ -d "cmd/$$cmd" ]; then \
			echo "Building $$cmd for all platforms..."; \
			GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o bin/$$cmd-darwin-arm64 ./cmd/$$cmd; \
			GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o bin/$$cmd-darwin-amd64 ./cmd/$$cmd; \
			GOOS=linux GOARCH=amd64 CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o bin/$$cmd-linux-amd64 ./cmd/$$cmd; \
			GOOS=linux GOARCH=arm64 CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o bin/$$cmd-linux-arm64 ./cmd/$$cmd; \
			GOOS=windows GOARCH=amd64 CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o bin/$$cmd-windows-amd64.exe ./cmd/$$cmd; \
		fi; \
	done
	@echo "Built binaries:"
	@ls -la bin/

# Run tests
test:
	$(GO) test -v -race ./...

# Run linter (requires golangci-lint)
lint:
	@command -v golangci-lint >/dev/null 2>&1 || { echo "Install: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; exit 1; }
	golangci-lint run ./...

# Clean build artifacts
clean:
	rm -rf bin/

# Install development tools
install-tools:
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Show help
help:
	@echo "Available targets:"
	@echo "  dev          - Build for current platform (fast)"
	@echo "  all          - Build for all platforms"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linter"
	@echo "  clean        - Remove built binaries"
	@echo "  install-tools - Install development tools"
	@echo ""
	@echo "Commands being built: $(CMDS)"
	@echo ""
	@echo "To add a new command, create cmd/<name>/main.go and add to CMDS"

# Quick test of hooks
test-hooks: dev
	@echo "Testing session-init..."
	@echo '{"session_id":"test","source":"startup"}' | HOOK_DEBUG=1 ./bin/session-init || true
