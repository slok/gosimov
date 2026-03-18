// Package write implements a [tool.Tool] that writes file contents.
//
// The tool writes content to a file, creating parent directories as needed.
// It reports whether the file was created or overwritten.
package write

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"

	"github.com/slok/gosimov/internal/utils/file"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

// Config configures the write [Tool].
type Config struct {
	// CWD is the working directory that paths are resolved relative to (required).
	CWD string
	// FS is the writable filesystem (optional).
	// If not set, defaults to OS operations rooted at CWD.
	FS file.ReadWriteFS
}

func (c *Config) defaults() error {
	if c.CWD == "" {
		return fmt.Errorf("cwd is required")
	}

	if c.FS == nil {
		c.FS = file.NewOSReadWriteFS(c.CWD)
	}

	return nil
}

// input is the JSON schema input for the write tool.
type input struct {
	Path    string `json:"path" jsonschema:"required,description=Path to the file to write, relative to working directory"`
	Content string `json:"content" jsonschema:"required,description=Content to write to the file"`
}

var inputSchema = toolschema.MustFromType[input]()

// Tool writes file contents.
type Tool struct {
	fsys file.ReadWriteFS
}

// New creates a new write tool.
func New(config Config) (*Tool, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid write tool config: %w", err)
	}

	return &Tool{fsys: config.FS}, nil
}

func (t *Tool) ID() string { return "write" }

func (t *Tool) Description() string {
	return "Writes content to a file. Creates the file if it doesn't exist, overwrites if it does. Parent directories are created automatically."
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

	// Check if the file already exists.
	_, statErr := t.fsys.Stat(resolved)
	existed := statErr == nil

	// Create parent directories.
	dir := filepath.Dir(resolved)
	if dir != "." {
		if err := t.fsys.MkdirAll(dir); err != nil {
			return nil, fmt.Errorf("failed to create directory: %s", dir)
		}
	}

	// Write the file.
	data := []byte(in.Content)
	if err := t.fsys.WriteFile(resolved, data); err != nil {
		return nil, fmt.Errorf("failed to write file: %s", in.Path)
	}

	// Report result.
	action := "Created"
	if existed {
		action = "Overwrote"
	}

	text := fmt.Sprintf("%s %s (%d bytes)", action, in.Path, len(data))

	return &tool.Result{
		Content: []model.ContentPart{model.NewContentText(text)},
	}, nil
}
