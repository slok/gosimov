package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

type prInlineCommentCreateInput struct {
	Path string `json:"path" jsonschema:"required,description=Changed file path where the inline comment should be posted"`
	Line int    `json:"line" jsonschema:"required,description=Right-side line number in the pull request diff"`
	Body string `json:"body" jsonschema:"required,description=Inline review comment body in markdown"`
}

var prInlineCommentCreateSchema = toolschema.MustFromType[prInlineCommentCreateInput]()

type prInlineCommentCreateTool struct {
	state *State
}

func NewPRInlineCommentCreateTool(state *State) tool.Tool {
	return &prInlineCommentCreateTool{state: state}
}

func (t *prInlineCommentCreateTool) ID() string {
	return "pr_inline_comment_create"
}

func (t *prInlineCommentCreateTool) Description() string {
	return "Create one inline PR review comment on an added/changed line"
}

func (t *prInlineCommentCreateTool) Schema() json.RawMessage {
	return prInlineCommentCreateSchema
}

func (t *prInlineCommentCreateTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	var in prInlineCommentCreateInput
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	in.Path = strings.TrimSpace(in.Path)
	in.Body = strings.TrimSpace(in.Body)
	if in.Path == "" {
		return nil, fmt.Errorf("path is required")
	}
	if in.Body == "" {
		return nil, fmt.Errorf("body is required")
	}
	if err := t.state.ValidateInlineTarget(in.Path, in.Line); err != nil {
		return nil, err
	}

	head, _, err := t.state.PRRefs(ctx)
	if err != nil {
		return nil, err
	}

	if t.state.gh.DryRun() {
		t.state.incrementInlineComments()
		return textResult(fmt.Sprintf("dry-run: would comment on %s:%d", in.Path, in.Line)), nil
	}

	endpoint := t.state.gh.PREndpoint("comments")
	_, err = t.state.gh.run(ctx,
		"api", "--method", "POST", endpoint,
		"-f", "body="+in.Body,
		"-f", "commit_id="+head,
		"-f", "path="+in.Path,
		"-F", "line="+strconv.Itoa(in.Line),
		"-f", "side=RIGHT",
	)
	if err != nil {
		t.state.decrementInlineComments()
		return nil, fmt.Errorf("creating inline comment: %w", err)
	}
	t.state.incrementInlineComments()

	return textResult(fmt.Sprintf("created inline comment on %s:%d", in.Path, in.Line)), nil
}
