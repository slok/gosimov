package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

type prFileReadInput struct {
	Path   string `json:"path" jsonschema:"required,description=Repository-relative file path to read"`
	Ref    string `json:"ref" jsonschema:"description=Either head or base (default head)"`
	Offset int    `json:"offset" jsonschema:"description=1-based line offset (default 1)"`
	Limit  int    `json:"limit" jsonschema:"description=Maximum lines to return (default 400, max 2000)"`
}

var prFileReadSchema = toolschema.MustFromType[prFileReadInput]()

type prFileReadTool struct {
	state *State
}

func NewPRFileReadTool(state *State) tool.Tool {
	return &prFileReadTool{state: state}
}

func (t *prFileReadTool) ID() string {
	return "pr_file_read"
}

func (t *prFileReadTool) Description() string {
	return "Read full file content from PR head or base revision"
}

func (t *prFileReadTool) Schema() json.RawMessage {
	return prFileReadSchema
}

func (t *prFileReadTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	var in prFileReadInput
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	in.Path = strings.TrimSpace(in.Path)
	if in.Path == "" {
		return nil, fmt.Errorf("path is required")
	}

	if in.Offset <= 0 {
		in.Offset = 1
	}
	if in.Limit <= 0 {
		in.Limit = 400
	}
	if in.Limit > 2000 {
		in.Limit = 2000
	}

	head, base, err := t.state.PRRefs(ctx)
	if err != nil {
		return nil, err
	}

	ref := head
	var refLabel string
	switch strings.ToLower(strings.TrimSpace(in.Ref)) {
	case "", "head":
		ref = head
		refLabel = "head"
	case "base":
		ref = base
		refLabel = "base"
	default:
		return nil, fmt.Errorf("ref must be one of: head, base")
	}

	endpoint := fmt.Sprintf("repos/%s/contents/%s?ref=%s", t.state.gh.Repo(), escapeContentPath(in.Path), url.QueryEscape(ref))
	b, err := t.state.gh.run(ctx, "api", endpoint, "-H", "Accept: application/vnd.github.raw+json")
	if err != nil {
		return nil, fmt.Errorf("reading file content from %s ref: %w", refLabel, err)
	}

	content := strings.ReplaceAll(string(b), "\r\n", "\n")
	lines := strings.Split(content, "\n")

	start := in.Offset - 1
	if start >= len(lines) {
		return nil, fmt.Errorf("offset %d exceeds file length (%d lines)", in.Offset, len(lines))
	}

	end := start + in.Limit
	if end > len(lines) {
		end = len(lines)
	}

	var out strings.Builder
	fmt.Fprintf(&out, "Path: %s\nRef: %s (%s)\nShowing file lines %d-%d of %d\n\n", in.Path, refLabel, ref[:12], start+1, end, len(lines))
	for i := start; i < end; i++ {
		fmt.Fprintf(&out, "%d: %s\n", i+1, lines[i])
	}
	if end < len(lines) {
		fmt.Fprintf(&out, "\n[%d more lines. Use offset=%d]", len(lines)-end, end+1)
	}

	return textResult(out.String()), nil
}

func escapeContentPath(path string) string {
	clean := filepath.ToSlash(strings.TrimSpace(path))
	parts := strings.Split(clean, "/")
	encoded := make([]string, 0, len(parts))
	for _, p := range parts {
		if p == "" {
			continue
		}
		encoded = append(encoded, url.PathEscape(p))
	}

	return strings.Join(encoded, "/")
}
