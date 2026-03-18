package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewSkillTool(t *testing.T) {
	tests := map[string]struct {
		prepare func(t *testing.T) SkillToolConfig
		expErr  bool
	}{
		"Missing skills dir should fail.": {
			prepare: func(t *testing.T) SkillToolConfig {
				t.Helper()
				return SkillToolConfig{}
			},
			expErr: true,
		},

		"Skills dir is a file should fail.": {
			prepare: func(t *testing.T) SkillToolConfig {
				t.Helper()

				p := filepath.Join(t.TempDir(), "not-a-dir")
				require.NoError(t, os.WriteFile(p, []byte("x"), 0o600))

				return SkillToolConfig{SkillsDir: p}
			},
			expErr: true,
		},

		"Invalid skill frontmatter should fail.": {
			prepare: func(t *testing.T) SkillToolConfig {
				t.Helper()

				root := t.TempDir()
				require.NoError(t, os.MkdirAll(filepath.Join(root, "broken"), 0o755))
				require.NoError(t, os.WriteFile(
					filepath.Join(root, "broken", "SKILL.md"),
					[]byte("---\nname: broken\n---\nbody"),
					0o600,
				))

				return SkillToolConfig{SkillsDir: root}
			},
			expErr: true,
		},

		"Duplicate names should fail.": {
			prepare: func(t *testing.T) SkillToolConfig {
				t.Helper()

				root := t.TempDir()
				writeSkillFile(t, root, "a", "same", "desc a", "body")
				writeSkillFile(t, root, "b", "same", "desc b", "body")

				return SkillToolConfig{SkillsDir: root}
			},
			expErr: true,
		},

		"Valid skills should load.": {
			prepare: func(t *testing.T) SkillToolConfig {
				t.Helper()

				root := t.TempDir()
				writeSkillFile(t, root, "one", "alpha", "alpha skill", "alpha body")

				return SkillToolConfig{SkillsDir: root}
			},
			expErr: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			_, err := NewSkillTool(test.prepare(t))

			if test.expErr {
				assert.Error(err)
				return
			}

			assert.NoError(err)
		})
	}
}

func TestSkillDescription(t *testing.T) {
	tests := map[string]struct {
		prepare func(t *testing.T) *skillTool
		assert  func(t *testing.T, description string)
	}{
		"No skills should report empty catalog.": {
			prepare: func(t *testing.T) *skillTool {
				t.Helper()

				root := t.TempDir()
				tool, err := NewSkillTool(SkillToolConfig{SkillsDir: root})
				require.NoError(t, err)

				return tool.(*skillTool)
			},
			assert: func(t *testing.T, description string) {
				t.Helper()
				assert := assert.New(t)

				assert.Contains(description, "No skills are currently available")
			},
		},

		"Skills should be listed in sorted order.": {
			prepare: func(t *testing.T) *skillTool {
				t.Helper()

				root := t.TempDir()
				writeSkillFile(t, root, "z", "zeta", "z desc", "z body")
				writeSkillFile(t, root, "a", "alpha", "a desc", "a body")

				tool, err := NewSkillTool(SkillToolConfig{SkillsDir: root})
				require.NoError(t, err)

				return tool.(*skillTool)
			},
			assert: func(t *testing.T, description string) {
				t.Helper()
				assert := assert.New(t)

				assert.Contains(description, "<available_skills>")
				assert.Contains(description, "<name>alpha</name>")
				assert.Contains(description, "<name>zeta</name>")

				alphaPos := strings.Index(description, "<name>alpha</name>")
				zetaPos := strings.Index(description, "<name>zeta</name>")
				assert.Greater(zetaPos, alphaPos)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			tool := test.prepare(t)
			description := tool.Description()
			test.assert(t, description)
		})
	}
}

func TestSkillExecute(t *testing.T) {
	root := t.TempDir()
	writeSkillFile(t, root, "release", "git-release", "prepare release notes", "## Steps\n- inspect commits")

	toolAny, err := NewSkillTool(SkillToolConfig{SkillsDir: root})
	require.NoError(t, err)
	tool := toolAny.(*skillTool)

	tests := map[string]struct {
		args   json.RawMessage
		expErr bool
		assert func(t *testing.T, output string)
	}{
		"Unknown argument should fail strict decode.": {
			args:   json.RawMessage(`{"name":"git-release","extra":true}`),
			expErr: true,
		},

		"Missing name should fail.": {
			args:   json.RawMessage(`{"name":"   "}`),
			expErr: true,
		},

		"Unknown skill should fail.": {
			args:   json.RawMessage(`{"name":"unknown"}`),
			expErr: true,
		},

		"Known skill should load skill content.": {
			args:   json.RawMessage(`{"name":"git-release"}`),
			expErr: false,
			assert: func(t *testing.T, output string) {
				t.Helper()
				assert := assert.New(t)

				assert.Contains(output, `<skill_content name="git-release">`)
				assert.Contains(output, "# Skill: git-release")
				assert.Contains(output, "## Steps")
				assert.Contains(output, "Base directory for this skill:")
				assert.Contains(output, "</skill_content>")
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			result, err := tool.Execute(context.Background(), test.args)

			if test.expErr {
				assert.Error(err)
				return
			}

			require.NoError(err)
			require.NotNil(result)
			require.Len(result.Content, 1)

			if test.assert != nil {
				test.assert(t, result.Content[0].Text)
			}
		})
	}
}

func writeSkillFile(t *testing.T, rootDir string, skillDir string, name string, description string, body string) string {
	t.Helper()

	path := filepath.Join(rootDir, skillDir, "SKILL.md")
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))

	content := strings.Join([]string{
		"---",
		"name: " + name,
		"description: " + description,
		"---",
		"",
		body,
		"",
	}, "\n")

	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))

	return path
}
