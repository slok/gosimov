// Package edit implements a [tool.Tool] that performs search-and-replace edits on files.
//
// The tool finds an exact text match in a file and replaces it. It supports
// CRLF normalization, uniqueness enforcement, optional replace-all mode,
// and optional mtime-based staleness detection.
package edit

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/slok/gosimov/internal/utils/file"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

// Config configures the edit [Tool].
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

// input is the JSON schema input for the edit tool.
type input struct {
	Path       string `json:"path" jsonschema:"required,description=Path to the file to edit, relative to working directory"`
	OldText    string `json:"old_text" jsonschema:"required,description=Exact text to find in the file"`
	NewText    string `json:"new_text" jsonschema:"required,description=Text to replace old_text with"`
	ReplaceAll bool   `json:"replace_all" jsonschema:"description=Replace all occurrences instead of requiring a unique match (default: false)"`
	Mtime      string `json:"mtime" jsonschema:"description=Expected file modification time from a previous read (RFC3339). If the file was modified since, the edit is rejected."`
}

var inputSchema = toolschema.MustFromType[input]()

// Tool performs search-and-replace edits on files.
type Tool struct {
	fsys file.ReadWriteFS
}

// New creates a new edit tool.
func New(config Config) (*Tool, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid edit tool config: %w", err)
	}

	return &Tool{fsys: config.FS}, nil
}

func (t *Tool) ID() string { return "edit" }

func (t *Tool) Description() string {
	return "Performs search-and-replace on a file. Finds old_text exactly and replaces it with new_text. " +
		"Optionally pass mtime (from a previous read) to detect if the file changed since you last read it."
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

	if in.OldText == "" {
		return nil, fmt.Errorf("old_text is required (use the write tool to create new files)")
	}

	if in.OldText == in.NewText {
		return nil, fmt.Errorf("old_text and new_text are identical, no changes to apply")
	}

	// Resolve and sanitize the path.
	resolved, err := file.SanitizePath(in.Path)
	if err != nil {
		return nil, err
	}

	// Check mtime staleness if provided.
	if in.Mtime != "" {
		if err := t.checkMtime(resolved, in.Mtime); err != nil {
			return nil, err
		}
	}

	// Read existing content.
	data, err := t.fsys.ReadFile(resolved)
	if err != nil {
		return nil, fmt.Errorf("file not found: %s", in.Path)
	}

	content := string(data)

	// Normalize CRLF to LF for matching.
	content, wasCRLF := normalizeCRLF(content)
	oldText := normalizeSearchText(in.OldText)
	newText := normalizeSearchText(in.NewText)

	// Perform the replacement and track positions for diff.
	var replaced string
	var replacements []file.Replacement

	if in.ReplaceAll {
		if !strings.Contains(content, oldText) {
			return nil, fmt.Errorf("old_text not found in %s — make sure it matches exactly, including whitespace and line breaks", in.Path)
		}

		replacements = findAllOccurrences(content, oldText, len(newText))
		replaced = strings.ReplaceAll(content, oldText, newText)
	} else {
		idx := strings.Index(content, oldText)
		if idx == -1 {
			return nil, fmt.Errorf("old_text not found in %s — make sure it matches exactly, including whitespace and line breaks", in.Path)
		}

		// Uniqueness check.
		lastIdx := strings.LastIndex(content, oldText)
		if idx != lastIdx {
			return nil, fmt.Errorf("old_text appears multiple times in %s — provide more surrounding context to make the match unique, or use replace_all", in.Path)
		}

		replacements = []file.Replacement{{Offset: idx, OldLen: len(oldText), NewLen: len(newText)}}
		replaced = content[:idx] + newText + content[idx+len(oldText):]
	}

	// Compute unified diff on normalized (LF) content before CRLF restoration.
	diff := file.FormatUnifiedDiff(in.Path, content, replaced, replacements, file.DefaultContextLines)

	// Restore CRLF if the original file used it.
	if wasCRLF {
		replaced = strings.ReplaceAll(replaced, "\n", "\r\n")
	}

	// Write the modified content back.
	if err := t.fsys.WriteFile(resolved, []byte(replaced)); err != nil {
		return nil, fmt.Errorf("failed to write file: %s", in.Path)
	}

	text := fmt.Sprintf("Edited %s successfully.\n\n%s", in.Path, diff)

	return &tool.Result{
		Content: []model.ContentPart{model.NewContentText(text)},
	}, nil
}

// checkMtime compares the file's current mtime against an expected value.
func (t *Tool) checkMtime(path string, expected string) error {
	expectedTime, err := time.Parse(time.RFC3339, expected)
	if err != nil {
		return fmt.Errorf("invalid mtime format (expected RFC3339): %s", expected)
	}

	info, err := t.fsys.Stat(path)
	if err != nil {
		return fmt.Errorf("cannot stat file for mtime check: %s", path)
	}

	actual := info.ModTime().Truncate(time.Second)
	expectedTime = expectedTime.Truncate(time.Second)

	if !actual.Equal(expectedTime) {
		return fmt.Errorf("file %s has been modified since last read (expected %s, actual %s) — read the file again before editing",
			path, expectedTime.Format(time.RFC3339), actual.Format(time.RFC3339))
	}

	return nil
}

// normalizeCRLF converts \r\n to \n and reports whether the conversion happened.
func normalizeCRLF(s string) (string, bool) {
	if strings.Contains(s, "\r\n") {
		return strings.ReplaceAll(s, "\r\n", "\n"), true
	}

	return s, false
}

// normalizeSearchText normalizes the LLM-provided search/replace text by converting
// \r\n to \n. This ensures matching works regardless of the LLM's line ending style.
func normalizeSearchText(s string) string {
	return strings.ReplaceAll(s, "\r\n", "\n")
}

// findAllOccurrences returns a [file.Replacement] for every occurrence of oldText in content.
func findAllOccurrences(content, oldText string, newLen int) []file.Replacement {
	var reps []file.Replacement

	start := 0
	for {
		idx := strings.Index(content[start:], oldText)
		if idx == -1 {
			break
		}

		reps = append(reps, file.Replacement{
			Offset: start + idx,
			OldLen: len(oldText),
			NewLen: newLen,
		})
		start += idx + len(oldText)
	}

	return reps
}
