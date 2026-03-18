// Package ls implements a [tool.Tool] that lists directory contents.
//
// The tool returns a sorted, newline-separated list of entries with a "/"
// suffix for directories. Output is truncated if it exceeds configured limits.
package ls

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"sort"
	"strings"

	"github.com/slok/gosimov/internal/utils/file"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

const (
	defaultEntryLimit = 500
	defaultMaxBytes   = 50 * 1024 // 50KB.
)

// Config configures the ls [Tool].
type Config struct {
	// CWD is the working directory that paths are resolved relative to (required).
	CWD string
	// FS is the filesystem to read from (optional).
	// If not set, defaults to [os.DirFS] rooted at CWD.
	FS fs.FS
	// EntryLimit is the maximum number of entries to return. 0 defaults to 500.
	EntryLimit int
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

	if c.EntryLimit <= 0 {
		c.EntryLimit = defaultEntryLimit
	}

	if c.MaxBytes <= 0 {
		c.MaxBytes = defaultMaxBytes
	}

	return nil
}

// input is the JSON schema input for the ls tool.
type input struct {
	Path  string `json:"path" jsonschema:"description=Directory to list, relative to working directory (default: current directory)"`
	Limit int    `json:"limit" jsonschema:"description=Maximum number of entries to return (default: 500)"`
}

var inputSchema = toolschema.MustFromType[input]()

// Tool lists directory contents.
type Tool struct {
	cwd        string
	fsys       fs.FS
	entryLimit int
	maxBytes   int
}

// New creates a new ls tool.
func New(config Config) (*Tool, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid ls tool config: %w", err)
	}

	return &Tool{
		cwd:        config.CWD,
		fsys:       config.FS,
		entryLimit: config.EntryLimit,
		maxBytes:   config.MaxBytes,
	}, nil
}

func (t *Tool) ID() string { return "ls" }

func (t *Tool) Description() string {
	return "Lists directory contents. Returns entries sorted alphabetically with '/' suffix for directories."
}

func (t *Tool) Schema() json.RawMessage {
	return inputSchema
}

func (t *Tool) Execute(_ context.Context, args json.RawMessage) (*tool.Result, error) {
	var in input
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	// Default path.
	if in.Path == "" {
		in.Path = "."
	}

	// Resolve and sanitize the path.
	resolved, err := file.SanitizePath(in.Path)
	if err != nil {
		return nil, err
	}

	// Validate the path exists and is a directory.
	info, err := fs.Stat(t.fsys, resolved)
	if err != nil {
		return nil, fmt.Errorf("path not found: %s", in.Path)
	}

	if !info.IsDir() {
		return nil, fmt.Errorf("not a directory: %s", in.Path)
	}

	// Read directory entries.
	entries, err := fs.ReadDir(t.fsys, resolved)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %w", err)
	}

	// Sort entries case-insensitively.
	sort.Slice(entries, func(i, j int) bool {
		return strings.ToLower(entries[i].Name()) < strings.ToLower(entries[j].Name())
	})

	// Determine effective entry limit.
	effectiveLimit := t.entryLimit
	if in.Limit > 0 && in.Limit < effectiveLimit {
		effectiveLimit = in.Limit
	}

	// Format entries.
	var lines []string
	entryLimitReached := false

	for i, entry := range entries {
		if i >= effectiveLimit {
			entryLimitReached = true

			break
		}

		name := entry.Name()
		if entry.IsDir() {
			name += "/"
		}

		lines = append(lines, name)
	}

	// Empty directory.
	if len(lines) == 0 {
		return textResult("(empty directory)"), nil
	}

	output := strings.Join(lines, "\n")

	// Truncate by bytes if needed.
	output, truncResult := file.TruncateHead(output, file.TruncateOpts{MaxBytes: t.maxBytes})

	// Build notices.
	var notices []string
	if entryLimitReached {
		notices = append(notices, fmt.Sprintf("%d entries limit reached", effectiveLimit))
	}

	if truncResult.Truncated {
		notices = append(notices, fmt.Sprintf("%s limit reached", file.FormatSize(t.maxBytes)))
	}

	if len(notices) > 0 {
		output += "\n\n[" + strings.Join(notices, ". ") + "]"
	}

	return textResult(output), nil
}

func textResult(text string) *tool.Result {
	return &tool.Result{
		Content: []model.ContentPart{model.NewContentText(text)},
	}
}
