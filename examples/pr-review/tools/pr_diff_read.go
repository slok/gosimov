package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

type prDiffReadInput struct {
	Path   string `json:"path" jsonschema:"required,description=Relative path of changed file in the pull request"`
	Offset int    `json:"offset" jsonschema:"description=1-based line offset inside diff output (default 1)"`
	Limit  int    `json:"limit" jsonschema:"description=Maximum diff lines to return (default 400, max 1000)"`
}

var prDiffReadSchema = toolschema.MustFromType[prDiffReadInput]()

type prDiffReadTool struct {
	state *State
}

func NewPRDiffReadTool(state *State) tool.Tool {
	return &prDiffReadTool{state: state}
}

func (t *prDiffReadTool) ID() string {
	return "pr_diff_read"
}

func (t *prDiffReadTool) Description() string {
	return "Read unified diff patch for a specific PR file"
}

func (t *prDiffReadTool) Schema() json.RawMessage {
	return prDiffReadSchema
}

func (t *prDiffReadTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	var in prDiffReadInput
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
	if in.Limit > 1000 {
		in.Limit = 1000
	}

	files, err := t.state.PRFiles(ctx)
	if err != nil {
		return nil, err
	}

	var match *prFile
	for _, file := range files {
		if file.Filename == in.Path {
			fileCopy := file
			match = &fileCopy
			break
		}
	}

	if match == nil {
		paths := make([]string, 0, len(files))
		for _, file := range files {
			paths = append(paths, file.Filename)
		}
		sort.Strings(paths)

		return nil, fmt.Errorf("path %q is not in changed files: %s", in.Path, strings.Join(paths, ", "))
	}

	patch := match.Patch
	if strings.TrimSpace(patch) == "" {
		patch = "(no textual patch available for this file)"
	}

	lines := strings.Split(patch, "\n")
	start := in.Offset - 1
	if start >= len(lines) {
		return nil, fmt.Errorf("offset %d exceeds diff length (%d lines)", in.Offset, len(lines))
	}

	end := start + in.Limit
	if end > len(lines) {
		end = len(lines)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "File: %s\nStatus: %s +%d/-%d\n", match.Filename, match.Status, match.Additions, match.Deletions)
	fmt.Fprintf(&b, "Showing diff lines %d-%d of %d\n\n", start+1, end, len(lines))
	for i := start; i < end; i++ {
		fmt.Fprintf(&b, "%d: %s\n", i+1, lines[i])
	}
	if end < len(lines) {
		fmt.Fprintf(&b, "\n[%d more diff lines. Use offset=%d]", len(lines)-end, end+1)
	}

	return textResult(b.String()), nil
}
