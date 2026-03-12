// Package shell implements a [tool.Tool] that executes shell commands.
//
// Each command spawns a new process — there is no persistent shell state.
// Output is truncated to configurable limits, keeping the tail (where
// errors and results typically appear).
//
// The [Executor] interface allows pluggable command execution backends
// (e.g., local exec, Docker, SSH). The default is [CMDExecutor], which
// spawns a new process per command via [exec.Command].
package shell

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/slok/gosimov/internal/utils/file"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

// Config configures the shell [Tool].
type Config struct {
	// CWD is the working directory for commands (required).
	CWD string
	// Executor is the shell executor to use (optional).
	// If not set, a default [CMDExecutor] is created.
	Executor Executor
	// DefaultTimeout is the per-command timeout when not specified by the LLM (optional).
	// If 0, defaults to [DefaultTimeout].
	DefaultTimeout time.Duration
	// MaxLines is the maximum number of output lines to return (optional).
	// If 0, defaults to 2000.
	MaxLines int
	// MaxBytes is the maximum bytes of output to return (optional).
	// If 0, defaults to 50KB.
	MaxBytes int
}

func (c *Config) defaults() error {
	if c.CWD == "" {
		return fmt.Errorf("cwd is required")
	}

	if c.Executor == nil {
		e, err := NewCMDExecutor(CMDExecutorConfig{})
		if err != nil {
			return fmt.Errorf("failed to create default executor: %w", err)
		}
		c.Executor = e
	}

	if c.DefaultTimeout <= 0 {
		c.DefaultTimeout = DefaultTimeout
	}

	if c.MaxLines <= 0 {
		c.MaxLines = 2000
	}

	if c.MaxBytes <= 0 {
		c.MaxBytes = 50 * 1024
	}

	return nil
}

// input is the JSON schema input for the shell tool.
type input struct {
	Command string `json:"command" jsonschema:"required,description=The shell command to execute"`
	Timeout int    `json:"timeout" jsonschema:"description=Timeout in seconds (optional, default 120)"`
}

var inputSchema = toolschema.MustFromType[input]()

// Tool executes shell commands via an [Executor].
type Tool struct {
	executor       Executor
	cwd            string
	defaultTimeout time.Duration
	maxLines       int
	maxBytes       int
}

// New creates a new shell tool.
func New(config Config) (*Tool, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid shell tool config: %w", err)
	}

	return &Tool{
		executor:       config.Executor,
		cwd:            config.CWD,
		defaultTimeout: config.DefaultTimeout,
		maxLines:       config.MaxLines,
		maxBytes:       config.MaxBytes,
	}, nil
}

func (t *Tool) ID() string { return "shell" }

func (t *Tool) Description() string {
	return "Execute a shell command. " +
		"Each command runs in a new process — state does not persist across calls. " +
		"Returns stdout and stderr. Output is truncated to the last 2000 lines or 50KB."
}

func (t *Tool) Schema() json.RawMessage {
	return inputSchema
}

func (t *Tool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	var in input
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	if in.Command == "" {
		return nil, fmt.Errorf("command is required")
	}

	// Determine timeout.
	timeout := t.defaultTimeout
	if in.Timeout > 0 {
		timeout = time.Duration(in.Timeout) * time.Second
	}

	// Execute the command.
	result, err := t.executor.Exec(ctx, in.Command, t.cwd, timeout)
	if err != nil {
		return nil, fmt.Errorf("failed to execute command: %w", err)
	}

	// Build the output text.
	text := t.formatOutput(result)

	return &tool.Result{
		Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: text}},
	}, nil
}

// formatOutput builds the text result from a shell execution.
func (t *Tool) formatOutput(result *Result) string {
	output := result.Output()

	if output == "" {
		output = "(no output)"
	}

	// Apply tail truncation.
	truncated, tr := file.TruncateTail(output, file.TruncateOpts{
		MaxBytes: t.maxBytes,
		MaxLines: t.maxLines,
	})

	var b strings.Builder
	b.WriteString(truncated)

	// Append notices.
	if result.TimedOut {
		fmt.Fprintf(&b, "\n\n[Command timed out]")
	}

	if result.ExitCode != 0 && !result.TimedOut {
		fmt.Fprintf(&b, "\n\n[Exit code: %d]", result.ExitCode)
	}

	if tr.Truncated {
		// Save full output to a temp file so the LLM can access it.
		fullOutputPath := ""
		if f, err := os.CreateTemp("", "gosimov-shell-*.log"); err == nil {
			_, _ = f.WriteString(output)
			_ = f.Close()
			fullOutputPath = f.Name()
		}

		fmt.Fprintf(&b, "\n[Showing last %d of %d lines", tr.KeptLines, tr.OriginalLines)
		if tr.KeptBytes < tr.OriginalBytes {
			fmt.Fprintf(&b, " (%s limit)", file.FormatSize(t.maxBytes))
		}
		if fullOutputPath != "" {
			fmt.Fprintf(&b, ". Full output: %s", fullOutputPath)
		}
		b.WriteString("]")
	}

	return b.String()
}
