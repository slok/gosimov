package shell

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// DefaultTimeout is the default command timeout.
const DefaultTimeout = 120 * time.Second

// Result is the output of a command execution.
type Result struct {
	// Stdout is the command's standard output.
	Stdout string
	// Stderr is the command's standard error.
	Stderr string
	// ExitCode is the command's exit code. -1 if the command was killed or timed out.
	ExitCode int
	// TimedOut is true if the command exceeded its timeout.
	TimedOut bool
}

// Output returns combined stdout and stderr, with stderr appended after a blank line if non-empty.
func (r *Result) Output() string {
	stdout := strings.TrimRight(r.Stdout, "\n")
	stderr := strings.TrimRight(r.Stderr, "\n")

	switch {
	case stdout == "" && stderr == "":
		return ""
	case stderr == "":
		return stdout
	case stdout == "":
		return stderr
	default:
		return stdout + "\n\n" + stderr
	}
}

// Executor runs shell commands. Each call is stateless — a new process is spawned per command.
type Executor interface {
	Exec(ctx context.Context, command string, cwd string, timeout time.Duration) (*Result, error)
}

// CMDExecutorConfig configures a [CMDExecutor].
type CMDExecutorConfig struct {
	// ShellPath is the shell binary to use (optional).
	// If empty, resolves to bash or sh.
	ShellPath string
	// Env is the environment for spawned processes (optional).
	// If nil, inherits the current process environment.
	Env []string
}

func (c *CMDExecutorConfig) defaults() error {
	if c.ShellPath == "" {
		c.ShellPath = resolveShell()
	}

	if c.Env == nil {
		c.Env = os.Environ()
	}

	return nil
}

// CMDExecutor implements [Executor] by spawning a new process per command via [exec.Command].
type CMDExecutor struct {
	shellPath string
	env       []string
}

// NewCMDExecutor creates a new [CMDExecutor].
func NewCMDExecutor(config CMDExecutorConfig) (*CMDExecutor, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid executor config: %w", err)
	}

	return &CMDExecutor{
		shellPath: config.ShellPath,
		env:       config.Env,
	}, nil
}

// Exec runs a command in a new shell process.
func (e *CMDExecutor) Exec(ctx context.Context, command string, cwd string, timeout time.Duration) (*Result, error) {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, e.shellPath, "-c", command)
	cmd.Dir = cwd
	cmd.Env = e.env
	// Prevent stdin reads from hanging.
	cmd.Stdin = nil
	// Use process group so we can kill all children on timeout.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	result := &Result{
		Stdout: stdout.String(),
		Stderr: stderr.String(),
	}

	if ctx.Err() != nil {
		result.TimedOut = true
		result.ExitCode = -1
		// Kill the process group to clean up children.
		if cmd.Process != nil {
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		}
		return result, nil
	}

	if err != nil {
		// Extract exit code from the error.
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
			return result, nil
		}

		return nil, fmt.Errorf("failed to execute command: %w", err)
	}

	return result, nil
}

// resolveShell finds the best available shell binary.
func resolveShell() string {
	candidates := []string{"/bin/bash", "/usr/bin/bash"}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}

	// Try PATH.
	if p, err := exec.LookPath("bash"); err == nil {
		return p
	}

	if p, err := exec.LookPath("sh"); err == nil {
		return p
	}

	return "/bin/sh"
}
