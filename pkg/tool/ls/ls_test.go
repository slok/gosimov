package ls_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"testing"
	"testing/fstest"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/tool/ls"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		config ls.Config
		expErr bool
	}{
		"Valid config should create a tool.": {
			config: ls.Config{CWD: "/tmp"},
		},

		"Missing CWD should return an error.": {
			config: ls.Config{},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := ls.New(test.config)

			if test.expErr {
				assert.Error(err)
				assert.Nil(tool)
			} else {
				assert.NoError(err)
				require.NotNil(tool)
			}
		})
	}
}

func TestToolMetadata(t *testing.T) {
	tests := map[string]struct {
		expID string
	}{
		"Should return correct tool ID, description and valid schema.": {
			expID: "ls",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := ls.New(ls.Config{CWD: "/tmp", FS: fstest.MapFS{}})
			require.NoError(err)

			assert.Equal(test.expID, tool.ID())
			assert.NotEmpty(tool.Description())
			assert.True(json.Valid(tool.Schema()))
		})
	}
}

func TestToolExecute(t *testing.T) {
	tests := map[string]struct {
		fsys     fstest.MapFS
		config   func(fsys fstest.MapFS) ls.Config
		args     json.RawMessage
		expText  string
		expErr   bool // Expect a Go error from Execute.
		contains []string
	}{
		"List root directory should return sorted entries.": {
			fsys: fstest.MapFS{
				"README.md":  &fstest.MapFile{},
				"go.mod":     &fstest.MapFile{},
				"main.go":    &fstest.MapFile{},
				"src/app.go": &fstest.MapFile{},
				".gitignore": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{}`),
			expText: ".gitignore\ngo.mod\nmain.go\nREADME.md\nsrc/",
		},

		"Default path should be current directory.": {
			fsys: fstest.MapFS{
				"file.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{}`),
			expText: "file.txt",
		},

		"Empty args should default to current directory.": {
			fsys: fstest.MapFS{
				"file.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    nil,
			expText: "file.txt",
		},

		"List subdirectory should return its entries.": {
			fsys: fstest.MapFS{
				"src/main.go":  &fstest.MapFile{},
				"src/utils.go": &fstest.MapFile{},
				"README.md":    &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{"path": "src"}`),
			expText: "main.go\nutils.go",
		},

		"Directories should have / suffix.": {
			fsys: fstest.MapFS{
				"dir1/file.txt": &fstest.MapFile{},
				"dir2/file.txt": &fstest.MapFile{},
				"file.txt":      &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{}`),
			expText: "dir1/\ndir2/\nfile.txt",
		},

		"Entries should be sorted case-insensitively.": {
			fsys: fstest.MapFS{
				"Zebra.txt":  &fstest.MapFile{},
				"apple.txt":  &fstest.MapFile{},
				"Banana.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{}`),
			expText: "apple.txt\nBanana.txt\nZebra.txt",
		},

		"Empty directory should return placeholder.": {
			fsys: fstest.MapFS{
				"empty": &fstest.MapFile{Mode: fs.ModeDir | 0o755},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{"path": "empty"}`),
			expText: "(empty directory)",
		},

		"Path not found should return error result.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "nonexistent"}`),
			expErr:   true,
			contains: []string{"not found"},
		},

		"File path (not directory) should return error result.": {
			fsys: fstest.MapFS{
				"file.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "file.txt"}`),
			expErr:   true,
			contains: []string{"not a directory"},
		},

		"Absolute path should return error result.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "/etc/passwd"}`),
			expErr:   true,
			contains: []string{"absolute paths are not allowed"},
		},

		"Path traversal with .. should return error result.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "../../etc"}`),
			expErr:   true,
			contains: []string{"escapes working directory"},
		},

		"Path traversal with hidden .. should return error result.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "foo/../../.."}`),
			expErr:   true,
			contains: []string{"escapes working directory"},
		},

		"Dot-dot alone should return error result.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": ".."}`),
			expErr:   true,
			contains: []string{"escapes working directory"},
		},

		"Invalid JSON args should return error result.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{invalid`),
			expErr:   true,
			contains: []string{"invalid arguments"},
		},

		"Entry limit should truncate and add notice.": {
			fsys: func() fstest.MapFS {
				m := fstest.MapFS{}
				for i := range 10 {
					m[fmt.Sprintf("file%02d.txt", i)] = &fstest.MapFile{}
				}

				return m
			}(),
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys, EntryLimit: 3}
			},
			args:     json.RawMessage(`{}`),
			contains: []string{"file00.txt", "file01.txt", "file02.txt", "3 entries limit reached"},
		},

		"User limit should cap below entry limit.": {
			fsys: func() fstest.MapFS {
				m := fstest.MapFS{}
				for i := range 10 {
					m[fmt.Sprintf("file%02d.txt", i)] = &fstest.MapFile{}
				}

				return m
			}(),
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys, EntryLimit: 100}
			},
			args:     json.RawMessage(`{"limit": 2}`),
			contains: []string{"file00.txt", "file01.txt", "2 entries limit reached"},
		},

		"Byte limit should truncate output and add notice.": {
			fsys: func() fstest.MapFS {
				m := fstest.MapFS{}
				// Each filename is ~20 chars, 50 files = ~1000 bytes.
				for i := range 50 {
					m[fmt.Sprintf("long-filename-%03d.txt", i)] = &fstest.MapFile{}
				}

				return m
			}(),
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys, MaxBytes: 100}
			},
			args:     json.RawMessage(`{}`),
			contains: []string{"long-filename-", "limit reached"},
		},

		"Dot path should work.": {
			fsys: fstest.MapFS{
				"file.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{"path": "."}`),
			expText: "file.txt",
		},

		"Path with dot-slash prefix should work.": {
			fsys: fstest.MapFS{
				"src/main.go": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{"path": "./src"}`),
			expText: "main.go",
		},

		"Nested subdirectory should resolve correctly.": {
			fsys: fstest.MapFS{
				"a/b/c/deep.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{"path": "a/b/c"}`),
			expText: "deep.txt",
		},

		"Relative path within cwd with .. that stays inside should work.": {
			fsys: fstest.MapFS{
				"file.txt":      &fstest.MapFile{},
				"sub/inner.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{"path": "sub/.."}`),
			expText: "file.txt\nsub/",
		},

		"Hidden files should be included.": {
			fsys: fstest.MapFS{
				".hidden":     &fstest.MapFile{},
				".config":     &fstest.MapFile{},
				"visible.txt": &fstest.MapFile{},
			},
			config: func(fsys fstest.MapFS) ls.Config {
				return ls.Config{CWD: "/project", FS: fsys}
			},
			args:    json.RawMessage(`{}`),
			expText: ".config\n.hidden\nvisible.txt",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			config := test.config(test.fsys)
			tool, err := ls.New(config)
			require.NoError(err)

			result, err := tool.Execute(context.Background(), test.args)

			if test.expErr {
				require.Error(err)
				assert.Nil(result)

				errText := err.Error()
				for _, substr := range test.contains {
					assert.Contains(errText, substr, "expected %q in error, got: %s", substr, errText)
				}

				return
			}

			require.NoError(err)
			require.NotNil(result)

			text := result.Content[0].Text

			if test.expText != "" {
				assert.Equal(test.expText, text)
			}

			for _, substr := range test.contains {
				assert.Contains(text, substr, "expected %q in output, got: %s", substr, text)
			}
		})
	}
}
