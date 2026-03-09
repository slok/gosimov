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
			assert := assert.New(t)
			require := require.New(t)

			e, err := shell.NewCMDExecutor(test.config)

			if test.expErr {
				assert.Error(err)
				assert.Nil(e)
			} else {
				assert.NoError(err)
				require.NotNil(e)
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
				assert := assert.New(t)
				assert.Equal("hello\n", r.Stdout)
				assert.Empty(r.Stderr)
				assert.Equal(0, r.ExitCode)
				assert.False(r.TimedOut)
			},
		},

		"Command with stderr.": {
			command: "echo err >&2",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert := assert.New(t)
				assert.Empty(r.Stdout)
				assert.Equal("err\n", r.Stderr)
				assert.Equal(0, r.ExitCode)
			},
		},

		"Non-zero exit code.": {
			command: "exit 42",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal(42, r.ExitCode)
			},
		},

		"Multi-line output.": {
			command: "echo -e 'line1\\nline2\\nline3'",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal("line1\nline2\nline3\n", r.Stdout)
				assert.Equal(0, r.ExitCode)
			},
		},

		"Working directory is respected.": {
			command: "pwd",
			expOut: func(t *testing.T, r *shell.Result) {
				t.Helper()
				assert := assert.New(t)
				assert.NotEmpty(strings.TrimSpace(r.Stdout))
				assert.Equal(0, r.ExitCode)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
			require.NoError(err)

			r, err := e.Exec(context.Background(), test.command, t.TempDir(), 0)
			require.NoError(err)
			require.NotNil(r)
			test.expOut(t, r)
		})
	}
}

func TestCMDExecutorExecBehavior(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Command exceeding timeout should be killed and flagged as timed out.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
				require.NoError(err)

				r, err := e.Exec(context.Background(), "sleep 30", t.TempDir(), 200*time.Millisecond)
				require.NoError(err)
				assert.True(r.TimedOut)
				assert.Equal(-1, r.ExitCode)
			},
		},

		"Context cancellation should kill the command and flag as timed out.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
				require.NoError(err)

				ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
				defer cancel()

				r, err := e.Exec(ctx, "sleep 30", t.TempDir(), 30*time.Second)
				require.NoError(err)
				assert.True(r.TimedOut)
				assert.Equal(-1, r.ExitCode)
			},
		},

		"Each execution should be stateless with no shared environment.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				e, err := shell.NewCMDExecutor(shell.CMDExecutorConfig{})
				require.NoError(err)

				dir := t.TempDir()
				_, err = e.Exec(context.Background(), "export MY_VAR=hello", dir, 0)
				require.NoError(err)

				r, err := e.Exec(context.Background(), "echo ${MY_VAR:-unset}", dir, 0)
				require.NoError(err)
				assert.Equal("unset\n", r.Stdout)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
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
			assert := assert.New(t)
			assert.Equal(test.exp, test.result.Output())
		})
	}
}
