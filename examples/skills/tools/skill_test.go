package tools_test

import (
	"context"
	"encoding/json"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	skilltools "github.com/slok/gosimov/examples/skills/tools"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewSkillTool(t *testing.T) {
	tests := map[string]struct {
		prepare func(t *testing.T) skilltools.SkillToolConfig
		expErr  bool
	}{
		"Missing source should fail.": {
			prepare: func(t *testing.T) skilltools.SkillToolConfig {
				t.Helper()
				return skilltools.SkillToolConfig{}
			},
			expErr: true,
		},

		"Skills dir is a file should fail.": {
			prepare: func(t *testing.T) skilltools.SkillToolConfig {
				t.Helper()

				p := filepath.Join(t.TempDir(), "not-a-dir")
				require.NoError(t, os.WriteFile(p, []byte("x"), 0o600))

				return skilltools.SkillToolConfig{SkillsDir: p}
			},
			expErr: true,
		},

		"Invalid skill frontmatter should fail.": {
			prepare: func(t *testing.T) skilltools.SkillToolConfig {
				t.Helper()

				return skilltools.SkillToolConfig{
					FS: skillMapFS(map[string]string{
						"broken": "---\nname: broken\n---\nbody",
					}),
				}
			},
			expErr: true,
		},

		"Duplicate names should fail.": {
			prepare: func(t *testing.T) skilltools.SkillToolConfig {
				t.Helper()

				return skilltools.SkillToolConfig{
					FS: skillMapFS(map[string]string{
						"a": skillDoc("same", "desc a", "body"),
						"b": skillDoc("same", "desc b", "body"),
					}),
				}
			},
			expErr: true,
		},

		"Valid skills should load from fs.FS.": {
			prepare: func(t *testing.T) skilltools.SkillToolConfig {
				t.Helper()

				return skilltools.SkillToolConfig{
					SkillsDir: "mem-skills",
					FS: skillMapFS(map[string]string{
						"one": skillDoc("alpha", "alpha skill", "alpha body"),
					}),
				}
			},
			expErr: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			loadedTool, err := skilltools.NewSkillTool(test.prepare(t))

			if test.expErr {
				assert.Error(err)
				return
			}

			assert.NoError(err)
			require.NotNil(t, loadedTool)
			assert.Equal("skill", loadedTool.ID())
		})
	}
}

func TestSkillDescription(t *testing.T) {
	tests := map[string]struct {
		cfg    skilltools.SkillToolConfig
		assert func(t *testing.T, description string)
	}{
		"No skills should report empty catalog.": {
			cfg: skilltools.SkillToolConfig{FS: fstest.MapFS{}},
			assert: func(t *testing.T, description string) {
				t.Helper()
				assert := assert.New(t)

				assert.Contains(description, "No skills are currently available")
			},
		},

		"Skills should be listed in sorted order.": {
			cfg: skilltools.SkillToolConfig{
				SkillsDir: "mem-skills",
				FS: skillMapFS(map[string]string{
					"z": skillDoc("zeta", "z desc", "z body"),
					"a": skillDoc("alpha", "a desc", "a body"),
				}),
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
			loadedTool := mustNewSkillTool(t, test.cfg)

			description := loadedTool.Description()
			test.assert(t, description)

			secondDescription := loadedTool.Description()
			assert.Equal(t, description, secondDescription)
		})
	}
}

func TestSkillExecute(t *testing.T) {
	loadedTool := mustNewSkillTool(t, skilltools.SkillToolConfig{
		SkillsDir: "mem-skills",
		FS: skillMapFS(map[string]string{
			"release": skillDoc("git-release", "prepare release notes", "## Steps\n- inspect commits"),
		}),
	})

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
				assert.Contains(output, "Base directory for this skill: mem-skills/release")
				assert.Contains(output, "</skill_content>")
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			result, err := loadedTool.Execute(context.Background(), test.args)

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

func mustNewSkillTool(t *testing.T, cfg skilltools.SkillToolConfig) tool.Tool {
	t.Helper()

	loadedTool, err := skilltools.NewSkillTool(cfg)
	require.NoError(t, err)

	return loadedTool
}

func skillMapFS(skills map[string]string) fstest.MapFS {
	files := fstest.MapFS{}
	for dir, content := range skills {
		skillPath := path.Join(dir, "SKILL.md")
		files[skillPath] = &fstest.MapFile{Data: []byte(content)}
	}

	return files
}

func skillDoc(name string, description string, body string) string {
	return strings.Join([]string{
		"---",
		"name: " + name,
		"description: " + description,
		"---",
		"",
		body,
		"",
	}, "\n")
}
