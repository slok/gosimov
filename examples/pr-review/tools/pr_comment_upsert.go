package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

type prCommentUpsertInput struct {
	Body string `json:"body" jsonschema:"required,description=Markdown body for summary PR comment"`
}

var prCommentUpsertSchema = toolschema.MustFromType[prCommentUpsertInput]()

type prCommentUpsertTool struct {
	state *State
}

func NewPRCommentUpsertTool(state *State) tool.Tool {
	return &prCommentUpsertTool{state: state}
}

func (t *prCommentUpsertTool) ID() string {
	return "pr_comment_upsert"
}

func (t *prCommentUpsertTool) Description() string {
	return "Create or update the single automated markdown summary comment on the PR"
}

func (t *prCommentUpsertTool) Schema() json.RawMessage {
	return prCommentUpsertSchema
}

func (t *prCommentUpsertTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	var in prCommentUpsertInput
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	body := strings.TrimSpace(in.Body)
	if body == "" {
		return nil, fmt.Errorf("body is required")
	}

	if !strings.Contains(body, t.state.reviewMarker) {
		body += "\n\n" + t.state.reviewMarker
	}

	viewer, err := t.state.viewer(ctx)
	if err != nil {
		return nil, err
	}

	comments, err := listIssueComments(ctx, t.state.gh)
	if err != nil {
		return nil, err
	}

	var existingID int
	for _, c := range comments {
		if strings.Contains(c.Body, t.state.reviewMarker) && strings.EqualFold(c.User.Login, viewer) {
			existingID = c.ID
		}
	}

	if t.state.gh.DryRun() {
		action := "create"
		if existingID > 0 {
			action = "update"
		}
		t.state.SetSummaryCommentAction(action + " (dry-run)")
		return textResult(fmt.Sprintf("dry-run: would %s summary comment", action)), nil
	}

	if existingID > 0 {
		endpoint := fmt.Sprintf("repos/%s/issues/comments/%d", t.state.gh.Repo(), existingID)
		if _, err := t.state.gh.run(ctx, "api", "--method", "PATCH", endpoint, "-f", "body="+body); err != nil {
			return nil, fmt.Errorf("updating summary comment: %w", err)
		}
		t.state.SetSummaryCommentAction("update")
		return textResult(fmt.Sprintf("updated summary comment id=%d", existingID)), nil
	}

	endpoint := fmt.Sprintf("repos/%s/issues/%d/comments", t.state.gh.Repo(), t.state.gh.PRNumber())
	if _, err := t.state.gh.run(ctx, "api", "--method", "POST", endpoint, "-f", "body="+body); err != nil {
		return nil, fmt.Errorf("creating summary comment: %w", err)
	}

	t.state.SetSummaryCommentAction("create")
	return textResult("created summary comment"), nil
}
