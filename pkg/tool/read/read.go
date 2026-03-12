// Package read implements a [tool.Tool] that reads file contents.
//
// The tool reads a file and returns its text content, optionally sliced by
// offset (1-indexed line number) and limit (number of lines). Output is
// truncated if it exceeds configured byte or line limits.
//
// Image files (PNG, JPEG, GIF, WebP) are detected by magic bytes and
// returned as [model.ImageData] content parts. Binary files are rejected
// with an error.
//
// All successful results include the file's modification time as a
// text notice in the content.
package read

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"strings"
	"time"

	"github.com/slok/gosimov/internal/utils/file"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

const (
	defaultMaxLines = 2000
	defaultMaxBytes = 50 * 1024 // 50KB.
)

// Config configures the read [Tool].
type Config struct {
	// CWD is the working directory that paths are resolved relative to (required).
	CWD string
	// FS is the filesystem to read from (optional).
	// If not set, defaults to [os.DirFS] rooted at CWD.
	FS fs.FS
	// MaxLines is the maximum number of lines to return. 0 defaults to 2000.
	MaxLines int
	// MaxBytes is the maximum byte size of the output. 0 defaults to 50KB.
	MaxBytes int
}

func (c *Config) defaults() error {
	if c.CWD == "" {
		return fmt.Errorf("cwd is required")
	}

	if c.FS == nil {
		c.FS = os.DirFS(c.CWD)
	}

	if c.MaxLines <= 0 {
		c.MaxLines = defaultMaxLines
	}

	if c.MaxBytes <= 0 {
		c.MaxBytes = defaultMaxBytes
	}

	return nil
}

// input is the JSON schema input for the read tool.
type input struct {
	Path   string `json:"path" jsonschema:"required,description=Path to the file to read, relative to working directory"`
	Offset int    `json:"offset" jsonschema:"description=Line number to start reading from (1-indexed, default: 1)"`
	Limit  int    `json:"limit" jsonschema:"description=Maximum number of lines to read"`
}

var inputSchema = toolschema.MustFromType[input]()

// Tool reads file contents.
type Tool struct {
	fsys     fs.FS
	maxLines int
	maxBytes int
}

// New creates a new read tool.
func New(config Config) (*Tool, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid read tool config: %w", err)
	}

	return &Tool{
		fsys:     config.FS,
		maxLines: config.MaxLines,
		maxBytes: config.MaxBytes,
	}, nil
}

func (t *Tool) ID() string { return "read" }

func (t *Tool) Description() string {
	return "Reads a file and returns its text content. Supports offset (1-indexed line number) and limit (number of lines) for partial reads."
}

func (t *Tool) Schema() json.RawMessage {
	return inputSchema
}

func (t *Tool) Execute(_ context.Context, args json.RawMessage) (*tool.Result, error) {
	var in input
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	if in.Path == "" {
		return nil, fmt.Errorf("path is required")
	}

	// Resolve and sanitize the path.
	resolved, err := file.SanitizePath(in.Path)
	if err != nil {
		return nil, err
	}

	// Stat the file for modification time.
	info, err := fs.Stat(t.fsys, resolved)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %s", in.Path)
	}

	mtime := info.ModTime()

	// Read the file.
	data, err := fs.ReadFile(t.fsys, resolved)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %s", in.Path)
	}

	// Detect content kind.
	detect := file.DetectContent(data)

	switch detect.Kind {
	case file.ContentKindImage:
		return t.imageResult(data, detect.MimeType, mtime), nil
	case file.ContentKindBinary:
		return nil, fmt.Errorf("cannot read binary file: %s", in.Path)
	default:
		return t.textResult(data, in, mtime)
	}
}

func (t *Tool) imageResult(data []byte, mimeType string, mtime time.Time) *tool.Result {
	return &tool.Result{
		Content: []model.ContentPart{
			{Type: model.ContentPartTypeText, Text: fmt.Sprintf("Read image file (%s, modified %s)", mimeType, mtime.Format(time.RFC3339))},
			{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: data, MimeType: mimeType}},
		},
	}
}

func (t *Tool) textResult(data []byte, in input, mtime time.Time) (*tool.Result, error) {
	content := string(data)
	allLines := strings.Split(content, "\n")
	totalLines := len(allLines)

	// Apply offset (1-indexed).
	startLine := 0
	if in.Offset > 0 {
		startLine = in.Offset - 1
	}

	if startLine >= totalLines {
		return nil, fmt.Errorf("offset %d is beyond end of file (%d lines total)", in.Offset, totalLines)
	}

	lines := allLines[startLine:]

	// Apply user limit.
	userLimitApplied := false
	if in.Limit > 0 && in.Limit < len(lines) {
		lines = lines[:in.Limit]
		userLimitApplied = true
	}

	sliced := strings.Join(lines, "\n")

	// Truncate by lines and bytes.
	output, truncResult := file.TruncateHead(sliced, file.TruncateOpts{
		MaxLines: t.maxLines,
		MaxBytes: t.maxBytes,
	})

	// Calculate the line range shown.
	shownStart := startLine + 1 // 1-indexed.
	shownEnd := startLine + truncResult.KeptLines

	// Build notices.
	var notices []string

	if truncResult.Truncated {
		notice := fmt.Sprintf("Showing lines %d-%d of %d", shownStart, shownEnd, totalLines)
		if truncResult.KeptBytes < truncResult.OriginalBytes {
			notice += fmt.Sprintf(" (%s limit)", file.FormatSize(t.maxBytes))
		}
		notice += fmt.Sprintf(". Use offset=%d to continue.", shownEnd+1)
		notices = append(notices, notice)
	} else if userLimitApplied {
		remaining := totalLines - (startLine + len(lines))
		if remaining > 0 {
			notices = append(notices, fmt.Sprintf("%d more lines in file. Use offset=%d to continue.", remaining, shownEnd+1))
		}
	}

	notices = append(notices, fmt.Sprintf("Modified: %s", mtime.Format(time.RFC3339)))

	output += "\n\n[" + strings.Join(notices, " ") + "]"

	return &tool.Result{
		Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: output}},
	}, nil
}
