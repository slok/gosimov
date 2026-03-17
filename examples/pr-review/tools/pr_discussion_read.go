package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

type prDiscussionReadInput struct {
	Limit int `json:"limit" jsonschema:"description=Maximum comments/reviews to include per section (default 30, max 100)"`
}

var prDiscussionReadSchema = toolschema.MustFromType[prDiscussionReadInput]()

type prDiscussionReadTool struct {
	state *State
}

func NewPRDiscussionReadTool(state *State) tool.Tool {
	return &prDiscussionReadTool{state: state}
}

func (t *prDiscussionReadTool) ID() string {
	return "pr_discussion_read"
}

func (t *prDiscussionReadTool) Description() string {
	return "Read PR discussion: issue comments, reviews, and review comments"
}

func (t *prDiscussionReadTool) Schema() json.RawMessage {
	return prDiscussionReadSchema
}

func (t *prDiscussionReadTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	var in prDiscussionReadInput
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	if in.Limit <= 0 {
		in.Limit = 30
	}
	if in.Limit > 100 {
		in.Limit = 100
	}

	var view struct {
		Comments []struct {
			Author struct {
				Login string `json:"login"`
			} `json:"author"`
			Body string `json:"body"`
		} `json:"comments"`
		Reviews []struct {
			Author struct {
				Login string `json:"login"`
			} `json:"author"`
			State string `json:"state"`
			Body  string `json:"body"`
		} `json:"reviews"`
	}

	if err := t.state.gh.runJSON(ctx, &view,
		"pr", "view", strconv.Itoa(t.state.gh.PRNumber()),
		"--repo", t.state.gh.Repo(),
		"--json", "comments,reviews",
	); err != nil {
		return nil, fmt.Errorf("reading discussion summary: %w", err)
	}

	reviewComments := make([]map[string]any, 0, in.Limit)
	for page := 1; ; page++ {
		endpoint := fmt.Sprintf("%s/comments?per_page=100&page=%d", t.state.gh.PREndpoint(""), page)
		var part []map[string]any
		if err := t.state.gh.runJSON(ctx, &part, "api", endpoint); err != nil {
			return nil, fmt.Errorf("reading review comments page %d: %w", page, err)
		}

		if len(part) == 0 {
			break
		}

		for _, item := range part {
			reviewComments = append(reviewComments, item)
			if len(reviewComments) >= in.Limit {
				break
			}
		}

		if len(reviewComments) >= in.Limit {
			break
		}
	}

	type entry struct {
		Author string `json:"author"`
		Body   string `json:"body"`
		State  string `json:"state,omitempty"`
		Path   string `json:"path,omitempty"`
		Line   int    `json:"line,omitempty"`
	}

	resp := struct {
		IssueComments []entry `json:"issue_comments"`
		Reviews       []entry `json:"reviews"`
		ReviewLines   []entry `json:"review_line_comments"`
	}{
		IssueComments: make([]entry, 0, in.Limit),
		Reviews:       make([]entry, 0, in.Limit),
		ReviewLines:   make([]entry, 0, in.Limit),
	}

	for i, c := range view.Comments {
		if i >= in.Limit {
			break
		}
		resp.IssueComments = append(resp.IssueComments, entry{Author: c.Author.Login, Body: c.Body})
	}

	for i, r := range view.Reviews {
		if i >= in.Limit {
			break
		}
		resp.Reviews = append(resp.Reviews, entry{Author: r.Author.Login, Body: r.Body, State: r.State})
	}

	for _, c := range reviewComments {
		line := intFromAny(c["line"])
		author := ""
		if u, ok := c["user"].(map[string]any); ok {
			author, _ = u["login"].(string)
		}
		path, _ := c["path"].(string)
		body, _ := c["body"].(string)
		resp.ReviewLines = append(resp.ReviewLines, entry{Author: author, Body: body, Path: path, Line: line})
	}

	b, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encoding discussion response: %w", err)
	}

	return textResult(string(b)), nil
}
