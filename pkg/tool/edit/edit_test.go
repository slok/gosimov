package edit_test

import (
	"context"
	"encoding/json"
	"io/fs"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool/edit"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		config edit.Config
		expErr bool
	}{
		"Valid config should create a tool.": {
			config: edit.Config{CWD: "/tmp"},
		},

		"Missing CWD should return an error.": {
			config: edit.Config{},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := edit.New(test.config)

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
			expID: "edit",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := edit.New(edit.Config{CWD: "/tmp", FS: newMemFS()})
			require.NoError(err)

			assert.Equal(test.expID, tool.ID())
			assert.NotEmpty(tool.Description())
			assert.True(json.Valid(tool.Schema()))
		})
	}
}

var testMtime = time.Date(2025, 6, 15, 10, 30, 0, 0, time.UTC)

func TestToolExecute(t *testing.T) {
	tests := map[string]struct {
		setup    func(*memFS)
		args     json.RawMessage
		expErr   bool
		contains []string
		expFile  string // Expected file content after edit.
	}{
		"Simple replacement should include diff.": {
			setup: func(mfs *memFS) {
				mfs.putFile("hello.txt", "hello world", testMtime)
			},
			args:     json.RawMessage(`{"path": "hello.txt", "old_text": "world", "new_text": "gopher"}`),
			contains: []string{"Edited", "hello.txt", "successfully", "--- hello.txt", "+++ hello.txt", "-hello world", "+hello gopher"},
			expFile:  "hello gopher",
		},

		"Multi-line replacement should include diff with context.": {
			setup: func(mfs *memFS) {
				mfs.putFile("code.go", "func main() {\n\tfmt.Println(\"hello\")\n}", testMtime)
			},
			args:     json.RawMessage(`{"path": "code.go", "old_text": "fmt.Println(\"hello\")", "new_text": "fmt.Println(\"goodbye\")"}`),
			contains: []string{"Edited", "code.go", "@@", "-\tfmt.Println(\"hello\")", "+\tfmt.Println(\"goodbye\")", " func main() {"},
			expFile:  "func main() {\n\tfmt.Println(\"goodbye\")\n}",
		},

		"Replace all occurrences should include diff.": {
			setup: func(mfs *memFS) {
				mfs.putFile("repeat.txt", "foo bar foo baz foo", testMtime)
			},
			args:     json.RawMessage(`{"path": "repeat.txt", "old_text": "foo", "new_text": "qux", "replace_all": true}`),
			contains: []string{"Edited", "repeat.txt", "-foo bar foo baz foo", "+qux bar qux baz qux"},
			expFile:  "qux bar qux baz qux",
		},

		"Delete text (empty new_text) should show only removed lines in diff.": {
			setup: func(mfs *memFS) {
				mfs.putFile("remove.txt", "keep this remove this keep this", testMtime)
			},
			args:     json.RawMessage(`{"path": "remove.txt", "old_text": " remove this", "new_text": ""}`),
			contains: []string{"Edited", "-keep this remove this keep this", "+keep this keep this"},
			expFile:  "keep this keep this",
		},

		"CRLF file should match and preserve line endings.": {
			setup: func(mfs *memFS) {
				mfs.putFile("crlf.txt", "line1\r\nline2\r\nline3", testMtime)
			},
			args:     json.RawMessage(`{"path": "crlf.txt", "old_text": "line2", "new_text": "replaced"}`),
			contains: []string{"Edited", "-line2", "+replaced"},
			expFile:  "line1\r\nreplaced\r\nline3",
		},

		"LLM sends CRLF in old_text but file uses LF.": {
			setup: func(mfs *memFS) {
				mfs.putFile("lf.txt", "line1\nline2\nline3", testMtime)
			},
			args:     json.RawMessage(`{"path": "lf.txt", "old_text": "line1\r\nline2", "new_text": "replaced"}`),
			contains: []string{"Edited", "-line1", "-line2", "+replaced"},
			expFile:  "replaced\nline3",
		},

		"Mtime check passes when matching.": {
			setup: func(mfs *memFS) {
				mfs.putFile("guarded.txt", "old content", testMtime)
			},
			args:     json.RawMessage(`{"path": "guarded.txt", "old_text": "old", "new_text": "new", "mtime": "2025-06-15T10:30:00Z"}`),
			contains: []string{"Edited", "@@"},
			expFile:  "new content",
		},

		"Mtime check fails when file was modified.": {
			setup: func(mfs *memFS) {
				mfs.putFile("guarded.txt", "old content", testMtime.Add(5*time.Minute))
			},
			args:     json.RawMessage(`{"path": "guarded.txt", "old_text": "old", "new_text": "new", "mtime": "2025-06-15T10:30:00Z"}`),
			expErr:   true,
			contains: []string{"modified since last read"},
		},

		"Invalid mtime format should return error.": {
			setup: func(mfs *memFS) {
				mfs.putFile("file.txt", "content", testMtime)
			},
			args:     json.RawMessage(`{"path": "file.txt", "old_text": "content", "new_text": "new", "mtime": "not-a-date"}`),
			expErr:   true,
			contains: []string{"invalid mtime format"},
		},

		"File not found should return error.": {
			args:     json.RawMessage(`{"path": "missing.txt", "old_text": "x", "new_text": "y"}`),
			expErr:   true,
			contains: []string{"file not found"},
		},

		"Old text not found should return error.": {
			setup: func(mfs *memFS) {
				mfs.putFile("file.txt", "actual content", testMtime)
			},
			args:     json.RawMessage(`{"path": "file.txt", "old_text": "nonexistent", "new_text": "new"}`),
			expErr:   true,
			contains: []string{"old_text not found"},
		},

		"Multiple matches without replace_all should return error.": {
			setup: func(mfs *memFS) {
				mfs.putFile("dup.txt", "foo bar foo", testMtime)
			},
			args:     json.RawMessage(`{"path": "dup.txt", "old_text": "foo", "new_text": "qux"}`),
			expErr:   true,
			contains: []string{"appears multiple times", "replace_all"},
		},

		"Empty old_text should return error.": {
			args:     json.RawMessage(`{"path": "file.txt", "old_text": "", "new_text": "new"}`),
			expErr:   true,
			contains: []string{"old_text is required"},
		},

		"Identical old_text and new_text should return error.": {
			args:     json.RawMessage(`{"path": "file.txt", "old_text": "same", "new_text": "same"}`),
			expErr:   true,
			contains: []string{"identical"},
		},

		"Missing path should return error.": {
			args:     json.RawMessage(`{"old_text": "a", "new_text": "b"}`),
			expErr:   true,
			contains: []string{"path is required"},
		},

		"Absolute path should return error.": {
			args:     json.RawMessage(`{"path": "/etc/passwd", "old_text": "a", "new_text": "b"}`),
			expErr:   true,
			contains: []string{"absolute paths are not allowed"},
		},

		"Path traversal should return error.": {
			args:     json.RawMessage(`{"path": "../../etc/passwd", "old_text": "a", "new_text": "b"}`),
			expErr:   true,
			contains: []string{"escapes working directory"},
		},

		"Invalid JSON args should return error.": {
			args:     json.RawMessage(`{invalid`),
			expErr:   true,
			contains: []string{"invalid arguments"},
		},

		"File in subdirectory should include diff with correct path.": {
			setup: func(mfs *memFS) {
				mfs.putFile("src/main.go", "package main", testMtime)
			},
			args:     json.RawMessage(`{"path": "src/main.go", "old_text": "main", "new_text": "foo"}`),
			contains: []string{"Edited", "src/main.go", "--- src/main.go", "+++ src/main.go"},
			expFile:  "package foo",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			mfs := newMemFS()
			if test.setup != nil {
				test.setup(mfs)
			}

			tool, err := edit.New(edit.Config{CWD: "/project", FS: mfs})
			require.NoError(err)

			result, err := tool.Execute(context.Background(), test.args)

			if test.expErr {
				require.Error(err)
				assert.Nil(result)

				errText := err.Error()
				for _, substr := range test.contains {
					assert.Contains(errText, substr)
				}

				return
			}

			require.NoError(err)
			require.NotNil(result)

			require.Len(result.Content, 1)
			assert.Equal(model.ContentPartTypeText, result.Content[0].Type)

			text := result.Content[0].Text
			for _, substr := range test.contains {
				assert.Contains(text, substr)
			}

			if test.expFile != "" {
				data, err := mfs.ReadFile(extractPath(t, test.args))
				require.NoError(err)
				assert.Equal(test.expFile, string(data))
			}
		})
	}
}

// extractPath extracts the "path" field from JSON args for verification.
func extractPath(t *testing.T, raw json.RawMessage) string {
	t.Helper()

	require := require.New(t)

	var m map[string]any
	require.NoError(json.Unmarshal(raw, &m))

	p, ok := m["path"].(string)
	require.True(ok)

	return p
}

// memFS is an in-memory implementation of [file.ReadWriteFS] for testing.
type memFS struct {
	mu    sync.Mutex
	files map[string]memFile
}

type memFile struct {
	data    []byte
	modTime time.Time
}

func newMemFS() *memFS {
	return &memFS{files: make(map[string]memFile)}
}

func (m *memFS) putFile(path string, content string, modTime time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.files[path] = memFile{data: []byte(content), modTime: modTime}
}

func (m *memFS) Stat(path string) (fs.FileInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	f, ok := m.files[path]
	if !ok {
		return nil, fs.ErrNotExist
	}

	return &memFileInfo{name: path, size: int64(len(f.data)), modTime: f.modTime}, nil
}

func (m *memFS) ReadFile(path string) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	f, ok := m.files[path]
	if !ok {
		return nil, fs.ErrNotExist
	}

	return f.data, nil
}

func (m *memFS) MkdirAll(_ string) error { return nil }

func (m *memFS) WriteFile(path string, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	existing, ok := m.files[path]
	modTime := time.Now()
	if ok {
		modTime = existing.modTime
	}

	m.files[path] = memFile{data: data, modTime: modTime}

	return nil
}

func (m *memFS) ReadDir(_ string) ([]fs.DirEntry, error) { panic("not used in edit tests") }
func (m *memFS) AppendFile(_ string, _ []byte) error     { panic("not used in edit tests") }

// memFileInfo implements [fs.FileInfo] minimally for testing.
type memFileInfo struct {
	name    string
	size    int64
	modTime time.Time
}

func (i *memFileInfo) Name() string       { return i.name }
func (i *memFileInfo) Size() int64        { return i.size }
func (i *memFileInfo) Mode() fs.FileMode  { return 0o644 }
func (i *memFileInfo) ModTime() time.Time { return i.modTime }
func (i *memFileInfo) IsDir() bool        { return false }
func (i *memFileInfo) Sys() any           { return nil }
