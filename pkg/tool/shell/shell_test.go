package shell_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool/shell"
	"github.com/slok/gosimov/pkg/tool/shell/shellmock"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		config shell.Config
		expErr bool
	}{
		"Valid config should create a tool.": {
			config: shell.Config{CWD: "/tmp"},
		},

		"Missing CWD should return an error.": {
			config: shell.Config{},
			expErr: true,
		},

		"Config with custom executor should use it.": {
			config: shell.Config{
				CWD:      "/tmp",
				Executor: &shellmock.MockExecutor{},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := shell.New(test.config)

			if test.expErr {
				assert.Error(err)
				assert.Nil(tool)
			} else {
				assert.NoError(err)
				require.NotNil(tool)
				assert.Equal("shell", tool.ID())
				assert.NotEmpty(tool.Description())
				assert.NotEmpty(tool.Schema())
			}
		})
	}
}

func TestExecute(t *testing.T) {
	tests := map[string]struct {
		args       map[string]any
		mock       func(m *shellmock.MockExecutor)
		expErr     bool
		expErrMsg  string
		expContent func(t *testing.T, parts []model.ContentPart)
	}{
		"Simple echo should return output.": {
			args: map[string]any{"command": "echo hello"},
			mock: func(m *shellmock.MockExecutor) {
				m.On("Exec", mock.Anything, "echo hello", "/work", 120*time.Second).
					Return(&shell.Result{Stdout: "hello\n"}, nil)
			},
			expContent: func(t *testing.T, parts []model.ContentPart) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(parts, 1)
				assert.Equal(model.ContentPartTypeText, parts[0].Type)
				assert.Equal("hello", parts[0].Text)
			},
		},

		"Command with stderr should include stderr.": {
			args: map[string]any{"command": "echo out && echo err >&2"},
			mock: func(m *shellmock.MockExecutor) {
				m.On("Exec", mock.Anything, "echo out && echo err >&2", "/work", 120*time.Second).
					Return(&shell.Result{Stdout: "out\n", Stderr: "err\n"}, nil)
			},
			expContent: func(t *testing.T, parts []model.ContentPart) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(parts, 1)
				assert.Contains(parts[0].Text, "out")
				assert.Contains(parts[0].Text, "err")
			},
		},

		"Non-zero exit code should append exit code notice.": {
			args: map[string]any{"command": "exit 42"},
			mock: func(m *shellmock.MockExecutor) {
				m.On("Exec", mock.Anything, "exit 42", "/work", 120*time.Second).
					Return(&shell.Result{ExitCode: 42}, nil)
			},
			expContent: func(t *testing.T, parts []model.ContentPart) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(parts, 1)
				assert.Contains(parts[0].Text, "[Exit code: 42]")
			},
		},

		"Timed out command should append timeout notice.": {
			args: map[string]any{"command": "sleep 30"},
			mock: func(m *shellmock.MockExecutor) {
				m.On("Exec", mock.Anything, "sleep 30", "/work", 120*time.Second).
					Return(&shell.Result{TimedOut: true, ExitCode: -1}, nil)
			},
			expContent: func(t *testing.T, parts []model.ContentPart) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(parts, 1)
				assert.Contains(parts[0].Text, "[Command timed out]")
				assert.NotContains(parts[0].Text, "[Exit code:")
			},
		},

		"Command with no output should return placeholder.": {
			args: map[string]any{"command": "true"},
			mock: func(m *shellmock.MockExecutor) {
				m.On("Exec", mock.Anything, "true", "/work", 120*time.Second).
					Return(&shell.Result{}, nil)
			},
			expContent: func(t *testing.T, parts []model.ContentPart) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(parts, 1)
				assert.Contains(parts[0].Text, "(no output)")
			},
		},

		"Custom timeout from args should be used.": {
			args: map[string]any{"command": "ls", "timeout": 10},
			mock: func(m *shellmock.MockExecutor) {
				m.On("Exec", mock.Anything, "ls", "/work", 10*time.Second).
					Return(&shell.Result{Stdout: "file.txt\n"}, nil)
			},
			expContent: func(t *testing.T, parts []model.ContentPart) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(parts, 1)
				assert.Contains(parts[0].Text, "file.txt")
			},
		},

		"Executor error should propagate.": {
			args: map[string]any{"command": "ls"},
			mock: func(m *shellmock.MockExecutor) {
				m.On("Exec", mock.Anything, "ls", "/work", 120*time.Second).
					Return(nil, fmt.Errorf("spawn failed"))
			},
			expErr:    true,
			expErrMsg: "spawn failed",
		},

		"Missing command should return an error.": {
			args:      map[string]any{},
			mock:      func(m *shellmock.MockExecutor) {},
			expErr:    true,
			expErrMsg: "command is required",
		},

		"Empty command should return an error.": {
			args:      map[string]any{"command": ""},
			mock:      func(m *shellmock.MockExecutor) {},
			expErr:    true,
			expErrMsg: "command is required",
		},

		"Nil args should return an error.": {
			mock:      func(m *shellmock.MockExecutor) {},
			expErr:    true,
			expErrMsg: "command is required",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			me := shellmock.NewMockExecutor(t)
			test.mock(me)

			tool, err := shell.New(shell.Config{
				CWD:      "/work",
				Executor: me,
			})
			require.NoError(err)

			var args json.RawMessage
			if test.args != nil {
				args, err = json.Marshal(test.args)
				require.NoError(err)
			}

			result, err := tool.Execute(context.Background(), args)

			if test.expErr {
				assert.Error(err)
				if test.expErrMsg != "" {
					assert.Contains(err.Error(), test.expErrMsg)
				}
				return
			}

			require.NoError(err)
			require.NotNil(result)
			test.expContent(t, result.Content)
		})
	}
}

func TestExecuteTruncation(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Output exceeding MaxLines should be tail-truncated with notice.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				lines := make([]string, 20)
				for i := range lines {
					lines[i] = fmt.Sprintf("line%d", i+1)
				}
				bigOutput := strings.Join(lines, "\n") + "\n"

				me := shellmock.NewMockExecutor(t)
				me.On("Exec", mock.Anything, "generate", "/work", 120*time.Second).
					Return(&shell.Result{Stdout: bigOutput}, nil)

				tool, err := shell.New(shell.Config{
					CWD:      "/work",
					Executor: me,
					MaxLines: 5,
				})
				require.NoError(err)

				args, _ := json.Marshal(map[string]any{"command": "generate"})
				result, err := tool.Execute(context.Background(), args)
				require.NoError(err)
				require.Len(result.Content, 1)

				text := result.Content[0].Text
				assert.Contains(text, "Showing last 5 of 20 lines")
				assert.Contains(text, "Full output:")
				assert.Contains(text, "line20")
				assert.NotContains(text, "line1\n")
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}
