package write_test

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
	"github.com/slok/gosimov/pkg/tool/write"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		config write.Config
		expErr bool
	}{
		"Valid config should create a tool.": {
			config: write.Config{CWD: "/tmp"},
		},

		"Missing CWD should return an error.": {
			config: write.Config{},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := write.New(test.config)

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
			expID: "write",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := write.New(write.Config{CWD: "/tmp", FS: newMemFS()})
			require.NoError(err)

			assert.Equal(test.expID, tool.ID())
			assert.NotEmpty(tool.Description())
			assert.True(json.Valid(tool.Schema()))
		})
	}
}

func TestToolExecute(t *testing.T) {
	tests := map[string]struct {
		setup    func(*memFS)
		args     json.RawMessage
		expErr   bool
		contains []string
		check    func(t *testing.T, mfs *memFS)
	}{
		"Write a new file.": {
			args:     json.RawMessage(`{"path": "hello.txt", "content": "hello world"}`),
			contains: []string{"Created", "hello.txt", "11 bytes"},
			check: func(t *testing.T, mfs *memFS) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal([]byte("hello world"), mfs.files["hello.txt"])
			},
		},

		"Overwrite an existing file.": {
			setup: func(mfs *memFS) {
				mfs.files["existing.txt"] = []byte("old content")
			},
			args:     json.RawMessage(`{"path": "existing.txt", "content": "new content"}`),
			contains: []string{"Overwrote", "existing.txt", "11 bytes"},
			check: func(t *testing.T, mfs *memFS) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal([]byte("new content"), mfs.files["existing.txt"])
			},
		},

		"Write file in subdirectory should create parent dirs.": {
			args:     json.RawMessage(`{"path": "src/main.go", "content": "package main"}`),
			contains: []string{"Created", "src/main.go", "12 bytes"},
			check: func(t *testing.T, mfs *memFS) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal([]byte("package main"), mfs.files["src/main.go"])
				assert.Contains(mfs.dirs, "src")
			},
		},

		"Write file in deeply nested directory.": {
			args:     json.RawMessage(`{"path": "a/b/c/file.txt", "content": "deep"}`),
			contains: []string{"Created", "a/b/c/file.txt", "4 bytes"},
			check: func(t *testing.T, mfs *memFS) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal([]byte("deep"), mfs.files["a/b/c/file.txt"])
				assert.Contains(mfs.dirs, "a/b/c")
			},
		},

		"Write empty content.": {
			args:     json.RawMessage(`{"path": "empty.txt", "content": ""}`),
			contains: []string{"Created", "empty.txt", "0 bytes"},
			check: func(t *testing.T, mfs *memFS) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal([]byte(""), mfs.files["empty.txt"])
			},
		},

		"Missing path should return error.": {
			args:     json.RawMessage(`{"content": "hello"}`),
			expErr:   true,
			contains: []string{"path is required"},
		},

		"Absolute path should return error.": {
			args:     json.RawMessage(`{"path": "/etc/passwd", "content": "hack"}`),
			expErr:   true,
			contains: []string{"absolute paths are not allowed"},
		},

		"Path traversal should return error.": {
			args:     json.RawMessage(`{"path": "../../etc/passwd", "content": "hack"}`),
			expErr:   true,
			contains: []string{"escapes working directory"},
		},

		"Invalid JSON args should return error.": {
			args:     json.RawMessage(`{invalid`),
			expErr:   true,
			contains: []string{"invalid arguments"},
		},

		"Unknown argument should return error.": {
			args:     json.RawMessage(`{"path":"hello.txt","content":"hello","unknown":true}`),
			expErr:   true,
			contains: []string{"invalid arguments", "unknown field"},
		},

		"Write to root-level file should not call MkdirAll.": {
			args:     json.RawMessage(`{"path": "root.txt", "content": "data"}`),
			contains: []string{"Created", "root.txt"},
			check: func(t *testing.T, mfs *memFS) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal([]byte("data"), mfs.files["root.txt"])
				assert.Empty(mfs.dirs)
			},
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

			tool, err := write.New(write.Config{CWD: "/project", FS: mfs})
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

			if test.check != nil {
				test.check(t, mfs)
			}
		})
	}
}

// memFS is an in-memory implementation of [file.ReadWriteFS] for testing.
type memFS struct {
	mu    sync.Mutex
	files map[string][]byte
	dirs  map[string]bool
}

func newMemFS() *memFS {
	return &memFS{
		files: make(map[string][]byte),
		dirs:  make(map[string]bool),
	}
}

func (m *memFS) Stat(path string) (fs.FileInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.files[path]; ok {
		return &memFileInfo{name: path, size: int64(len(m.files[path]))}, nil
	}

	return nil, fs.ErrNotExist
}

func (m *memFS) ReadFile(path string) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	data, ok := m.files[path]
	if !ok {
		return nil, fs.ErrNotExist
	}

	return data, nil
}

func (m *memFS) MkdirAll(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.dirs[path] = true

	return nil
}

func (m *memFS) WriteFile(path string, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.files[path] = data

	return nil
}

func (m *memFS) ReadDir(_ string) ([]fs.DirEntry, error) { panic("not used in write tests") }
func (m *memFS) AppendFile(_ string, _ []byte) error     { panic("not used in write tests") }

// memFileInfo implements [fs.FileInfo] minimally for testing.
type memFileInfo struct {
	name string
	size int64
}

func (i *memFileInfo) Name() string       { return i.name }
func (i *memFileInfo) Size() int64        { return i.size }
func (i *memFileInfo) Mode() fs.FileMode  { return 0o644 }
func (i *memFileInfo) ModTime() time.Time { return time.Time{} }
func (i *memFileInfo) IsDir() bool        { return false }
func (i *memFileInfo) Sys() any           { return nil }
