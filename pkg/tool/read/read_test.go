package read_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool/read"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		config read.Config
		expErr bool
	}{
		"Valid config should create a tool.": {
			config: read.Config{CWD: "/tmp"},
		},

		"Missing CWD should return an error.": {
			config: read.Config{},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := read.New(test.config)

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
			expID: "read",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := read.New(read.Config{CWD: "/tmp", FS: fstest.MapFS{}})
			require.NoError(err)

			assert.Equal(test.expID, tool.ID())
			assert.NotEmpty(tool.Description())
			assert.True(json.Valid(tool.Schema()))
		})
	}
}

var testMtime = time.Date(2025, 6, 15, 10, 30, 0, 0, time.UTC)

func TestToolExecuteText(t *testing.T) {
	tests := map[string]struct {
		fsys     fstest.MapFS
		config   func(fsys fstest.MapFS) read.Config
		args     json.RawMessage
		expErr   bool
		contains []string
	}{
		"Read a simple file.": {
			fsys: fstest.MapFS{
				"hello.txt": &fstest.MapFile{Data: []byte("hello world"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "hello.txt"}`),
			contains: []string{"hello world", "Modified: 2025-06-15T10:30:00Z"},
		},

		"Read a multi-line file.": {
			fsys: fstest.MapFS{
				"lines.txt": &fstest.MapFile{Data: []byte("line1\nline2\nline3"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "lines.txt"}`),
			contains: []string{"line1\nline2\nline3", "Modified:"},
		},

		"Read file in subdirectory.": {
			fsys: fstest.MapFS{
				"src/main.go": &fstest.MapFile{Data: []byte("package main"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "src/main.go"}`),
			contains: []string{"package main", "Modified:"},
		},

		"Offset should start from the given line (1-indexed).": {
			fsys: fstest.MapFS{
				"lines.txt": &fstest.MapFile{Data: []byte("line1\nline2\nline3\nline4\nline5"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "lines.txt", "offset": 3}`),
			contains: []string{"line3\nline4\nline5", "Modified:"},
		},

		"Limit should cap the number of lines returned.": {
			fsys: fstest.MapFS{
				"lines.txt": &fstest.MapFile{Data: []byte("line1\nline2\nline3\nline4\nline5"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "lines.txt", "limit": 2}`),
			contains: []string{"line1\nline2", "3 more lines in file. Use offset=3 to continue.", "Modified:"},
		},

		"Offset and limit together.": {
			fsys: fstest.MapFS{
				"lines.txt": &fstest.MapFile{Data: []byte("line1\nline2\nline3\nline4\nline5"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "lines.txt", "offset": 2, "limit": 2}`),
			contains: []string{"line2\nline3", "2 more lines in file. Use offset=4 to continue.", "Modified:"},
		},

		"Offset at last line should return that line.": {
			fsys: fstest.MapFS{
				"lines.txt": &fstest.MapFile{Data: []byte("line1\nline2\nline3"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "lines.txt", "offset": 3}`),
			contains: []string{"line3", "Modified:"},
		},

		"Offset beyond end of file should return error.": {
			fsys: fstest.MapFS{
				"short.txt": &fstest.MapFile{Data: []byte("one\ntwo"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "short.txt", "offset": 10}`),
			expErr:   true,
			contains: []string{"beyond end of file"},
		},

		"File not found should return error.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "nonexistent.txt"}`),
			expErr:   true,
			contains: []string{"failed to read file"},
		},

		"Missing path should return error.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{}`),
			expErr:   true,
			contains: []string{"path is required"},
		},

		"Absolute path should return error.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "/etc/passwd"}`),
			expErr:   true,
			contains: []string{"absolute paths are not allowed"},
		},

		"Path traversal should return error.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "../../etc/passwd"}`),
			expErr:   true,
			contains: []string{"escapes working directory"},
		},

		"Invalid JSON args should return error.": {
			fsys: fstest.MapFS{},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{invalid`),
			expErr:   true,
			contains: []string{"invalid arguments"},
		},

		"Line limit truncation should add notice.": {
			fsys: func() fstest.MapFS {
				lines := make([]string, 100)
				for i := range lines {
					lines[i] = fmt.Sprintf("line%d", i+1)
				}

				return fstest.MapFS{
					"big.txt": &fstest.MapFile{Data: []byte(strings.Join(lines, "\n")), ModTime: testMtime},
				}
			}(),
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys, MaxLines: 5}
			},
			args:     json.RawMessage(`{"path": "big.txt"}`),
			contains: []string{"line1", "Showing lines 1-5 of 100", "Use offset=6 to continue.", "Modified:"},
		},

		"Byte limit truncation should add notice.": {
			fsys: func() fstest.MapFS {
				lines := make([]string, 50)
				for i := range lines {
					lines[i] = strings.Repeat("x", 100)
				}

				return fstest.MapFS{
					"big.txt": &fstest.MapFile{Data: []byte(strings.Join(lines, "\n")), ModTime: testMtime},
				}
			}(),
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys, MaxBytes: 500}
			},
			args:     json.RawMessage(`{"path": "big.txt"}`),
			contains: []string{"limit", "Use offset=", "Modified:"},
		},

		"Empty file should return mtime notice.": {
			fsys: fstest.MapFS{
				"empty.txt": &fstest.MapFile{Data: []byte(""), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "empty.txt"}`),
			contains: []string{"Modified: 2025-06-15T10:30:00Z"},
		},

		"Limit that covers entire file should still have mtime.": {
			fsys: fstest.MapFS{
				"short.txt": &fstest.MapFile{Data: []byte("one\ntwo"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "short.txt", "limit": 100}`),
			contains: []string{"one\ntwo", "Modified:"},
		},

		"Limit exactly matching remaining lines should still have mtime.": {
			fsys: fstest.MapFS{
				"lines.txt": &fstest.MapFile{Data: []byte("line1\nline2\nline3"), ModTime: testMtime},
			},
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys}
			},
			args:     json.RawMessage(`{"path": "lines.txt", "limit": 3}`),
			contains: []string{"line1\nline2\nline3", "Modified:"},
		},

		"Offset with limit and truncation should show correct range.": {
			fsys: func() fstest.MapFS {
				lines := make([]string, 20)
				for i := range lines {
					lines[i] = fmt.Sprintf("line%d", i+1)
				}

				return fstest.MapFS{
					"lines.txt": &fstest.MapFile{Data: []byte(strings.Join(lines, "\n")), ModTime: testMtime},
				}
			}(),
			config: func(fsys fstest.MapFS) read.Config {
				return read.Config{CWD: "/project", FS: fsys, MaxLines: 3}
			},
			args:     json.RawMessage(`{"path": "lines.txt", "offset": 5}`),
			contains: []string{"line5", "line6", "line7", "Showing lines 5-7 of 20", "Use offset=8 to continue.", "Modified:"},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			config := test.config(test.fsys)
			tool, err := read.New(config)
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

			// Text results should have a single text content part.
			require.Len(result.Content, 1)
			assert.Equal(model.ContentPartTypeText, result.Content[0].Type)

			text := result.Content[0].Text
			for _, substr := range test.contains {
				assert.Contains(text, substr)
			}
		})
	}
}

func TestToolExecuteImage(t *testing.T) {
	tests := map[string]struct {
		fsys    fstest.MapFS
		args    json.RawMessage
		expMime string
	}{
		"PNG image should return image content part.": {
			fsys: fstest.MapFS{
				"photo.png": &fstest.MapFile{Data: encodePNG(t), ModTime: testMtime},
			},
			args:    json.RawMessage(`{"path": "photo.png"}`),
			expMime: "image/png",
		},

		"JPEG image should return image content part.": {
			fsys: fstest.MapFS{
				"photo.jpg": &fstest.MapFile{Data: encodeJPEG(t), ModTime: testMtime},
			},
			args:    json.RawMessage(`{"path": "photo.jpg"}`),
			expMime: "image/jpeg",
		},

		"GIF image should return image content part.": {
			fsys: fstest.MapFS{
				"anim.gif": &fstest.MapFile{Data: encodeGIF(t), ModTime: testMtime},
			},
			args:    json.RawMessage(`{"path": "anim.gif"}`),
			expMime: "image/gif",
		},

		"WebP image should return image content part.": {
			fsys: fstest.MapFS{
				"photo.webp": &fstest.MapFile{Data: fakeWebP(), ModTime: testMtime},
			},
			args:    json.RawMessage(`{"path": "photo.webp"}`),
			expMime: "image/webp",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := read.New(read.Config{CWD: "/project", FS: test.fsys})
			require.NoError(err)

			result, err := tool.Execute(context.Background(), test.args)
			require.NoError(err)
			require.NotNil(result)

			// Should have two content parts: text note + image data.
			require.Len(result.Content, 2)

			// First part: text note with mime type and mtime.
			assert.Equal(model.ContentPartTypeText, result.Content[0].Type)
			assert.Contains(result.Content[0].Text, test.expMime)
			assert.Contains(result.Content[0].Text, "modified")

			// Second part: image data.
			assert.Equal(model.ContentPartTypeImage, result.Content[1].Type)
			require.NotNil(result.Content[1].Image)
			assert.Equal(test.expMime, result.Content[1].Image.MimeType)
			assert.NotEmpty(result.Content[1].Image.Data)
		})
	}
}

func TestToolExecuteBinary(t *testing.T) {
	tests := map[string]struct {
		fsys     fstest.MapFS
		args     json.RawMessage
		contains []string
	}{
		"Binary file with null bytes should be rejected.": {
			fsys: fstest.MapFS{
				"program.bin": &fstest.MapFile{
					Data:    []byte{0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00},
					ModTime: testMtime,
				},
			},
			args:     json.RawMessage(`{"path": "program.bin"}`),
			contains: []string{"cannot read binary file"},
		},

		"Binary file with embedded nulls should be rejected.": {
			fsys: fstest.MapFS{
				"data.dat": &fstest.MapFile{
					Data:    []byte("some text\x00more data\x00end"),
					ModTime: testMtime,
				},
			},
			args:     json.RawMessage(`{"path": "data.dat"}`),
			contains: []string{"cannot read binary file"},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			tool, err := read.New(read.Config{CWD: "/project", FS: test.fsys})
			require.NoError(err)

			result, err := tool.Execute(context.Background(), test.args)
			require.Error(err)
			assert.Nil(result)

			errText := err.Error()
			for _, substr := range test.contains {
				assert.Contains(errText, substr)
			}
		})
	}
}

// encodePNG creates a minimal valid PNG image.
func encodePNG(t *testing.T) []byte {
	t.Helper()

	require := require.New(t)

	img := image.NewRGBA(image.Rect(0, 0, 1, 1))

	var buf bytes.Buffer
	err := png.Encode(&buf, img)
	require.NoError(err, "failed to encode PNG")

	return buf.Bytes()
}

// encodeJPEG creates a minimal valid JPEG image.
func encodeJPEG(t *testing.T) []byte {
	t.Helper()

	require := require.New(t)

	img := image.NewRGBA(image.Rect(0, 0, 1, 1))

	var buf bytes.Buffer
	err := jpeg.Encode(&buf, img, nil)
	require.NoError(err, "failed to encode JPEG")

	return buf.Bytes()
}

// encodeGIF creates a minimal valid GIF image.
func encodeGIF(t *testing.T) []byte {
	t.Helper()

	require := require.New(t)

	img := image.NewPaletted(image.Rect(0, 0, 1, 1), color.Palette{color.Black})

	var buf bytes.Buffer
	err := gif.Encode(&buf, img, nil)
	require.NoError(err, "failed to encode GIF")

	return buf.Bytes()
}

// fakeWebP creates a minimal byte sequence with the WebP magic signature.
func fakeWebP() []byte {
	data := make([]byte, 20)
	copy(data[0:4], "RIFF")
	copy(data[8:12], "WEBP")

	return data
}
