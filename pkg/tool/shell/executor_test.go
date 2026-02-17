package shell_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/tool/shell"
)

func TestNewCMDExecutor(t *testing.T) {
	tests := map[string]struct {
		config shell.CMDExecutorConfig
		expErr bool
	}{
		"Default config should create an executor.": {
			config: shell.CMDExecutorConfig{},
		},

		"Custom shell path should be accepted.": {
			config: shell.CMDExecutorConfig{ShellPath: "/bin/sh"},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			e, err := shell.NewCMDExecutor(test.config)

			if test.expErr {
				assert.Error(t, err)
				assert.Nil(t, e)
			} else {
				assert.NoError(t, err)
				require.NotNil(t, e)
			}
		})
	}
}

func TestCMDExecutorExec(t *testing.T) {
	tests := map[string]struct {
		command string
		expOut  func(t *testing.T, r *shell.Result)
	}{
		"Simple echo.": {
			command: "echo hello",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert.Equal(t, "hello\n", r.Stdout)
				assert.Empty(t, r.Stderr)
				assert.Equal(t, 0, r.ExitCode)
				assert.False(t, r.TimedOut)
			},
		},

		"Command with stderr.": {
			command: "echo err >&2",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert.Empty(t, r.Stdout)
				assert.Equal(t, "err\n", r.Stderr)
				assert.Equal(t, 0, r.ExitCode)
			},
		},

		"Non-zero exit code.": {
			command: "exit 42",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert.Equal(t, 42, r.ExitCode)
			},
		},

		"Multi-line output.": {
			command: "echo -e 'line1\\nline2\\nline3'",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert.Equal(t, "line1\nline2\nline3\n", r.Stdout)
				assert.Equal(t, 0, r.ExitCode)
			},
		},

		"Working directory is respected.": {
			command: "pwd",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert.NotEmpty(t, strings.TrimSpace(r.Stdout))
				assert.Equal(t, 0, r.ExitCode)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
			require.NoError(t, err)

			r, err := e.Exec(context.Background(), test.command, t.TempDir(), 0)
			require.NoError(t, err)
			require.NotNil(t, r)
			test.expOut(t, r)
		})
	}
}

func TestCMDExecutorExecTimeout(t *testing.T) {
	e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
	require.NoError(t, err)

	r, err := e.Exec(context.Background(), "sleep 30", t.TempDir(), 200*time.Millisecond)
	require.NoError(t, err)
	assert.True(t, r.TimedOut)
	assert.Equal(t, -1, r.ExitCode)
}

func TestCMDExecutorExecContextCancellation(t *testing.T) {
	e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	r, err := e.Exec(ctx, "sleep 30", t.TempDir(), 30*time.Second)
	require.NoError(t, err)
	assert.True(t, r.TimedOut)
	assert.Equal(t, -1, r.ExitCode)
}

func TestCMDExecutorExecStatelessness(t *testing.T) {
	e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
	require.NoError(t, err)

	dir := t.TempDir()

	// Set env var in first command — should NOT persist to second.
	_, err = e.Exec(context.Background(), "export MY_VAR=hello", dir, 0)
	require.NoError(t, err)

	r, err := e.Exec(context.Background(), "echo ${MY_VAR:-unset}", dir, 0)
	require.NoError(t, err)
	assert.Equal(t, "unset\n", r.Stdout)
}

func TestResultOutput(t *testing.T) {
	tests := map[string]struct {
		result shell.Result
		exp    string
	}{
		"Only stdout.": {
			result: shell.Result{Stdout: "hello\n"},
			exp:    "hello",
		},

		"Only stderr.": {
			result: shell.Result{Stderr: "error\n"},
			exp:    "error",
		},

		"Both stdout and stderr.": {
			result: shell.Result{Stdout: "out\n", Stderr: "err\n"},
			exp:    "out\n\nerr",
		},

		"Empty output.": {
			result: shell.Result{},
			exp:    "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, test.result.Output())
		})
	}
}
