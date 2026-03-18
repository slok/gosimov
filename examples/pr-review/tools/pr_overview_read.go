package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

type noInput struct{}

var prOverviewReadSchema = toolschema.MustFromType[noInput]()

type prOverviewReadTool struct {
	state *State
}

func NewPROverviewReadTool(state *State) tool.Tool {
	return &prOverviewReadTool{state: state}
}

func (t *prOverviewReadTool) ID() string {
	return "pr_overview_read"
}

func (t *prOverviewReadTool) Description() string {
	return "Read PR metadata, commits, and changed files summary"
}

func (t *prOverviewReadTool) Schema() json.RawMessage {
	return prOverviewReadSchema
}

func (t *prOverviewReadTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	var in noInput
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	head, base, err := t.state.PRRefs(ctx)
	if err != nil {
		return nil, err
	}

	var prView struct {
		Number       int    `json:"number"`
		Title        string `json:"title"`
		Body         string `json:"body"`
		URL          string `json:"url"`
		BaseRefName  string `json:"baseRefName"`
		HeadRefName  string `json:"headRefName"`
		ChangedFiles int    `json:"changedFiles"`
		Additions    int    `json:"additions"`
		Deletions    int    `json:"deletions"`
		IsDraft      bool   `json:"isDraft"`
		Author       struct {
			Login string `json:"login"`
		} `json:"author"`
		Commits []struct {
			OID     string `json:"oid"`
			Message string `json:"messageHeadline"`
		} `json:"commits"`
	}

	if err := t.state.gh.runJSON(ctx, &prView,
		"pr", "view", strconv.Itoa(t.state.gh.PRNumber()),
		"--repo", t.state.gh.Repo(),
		"--json", "number,title,body,url,baseRefName,headRefName,changedFiles,additions,deletions,isDraft,author,commits",
	); err != nil {
		return nil, fmt.Errorf("reading PR overview: %w", err)
	}

	files, err := t.state.PRFiles(ctx)
	if err != nil {
		return nil, err
	}

	type fileSummary struct {
		Path      string `json:"path"`
		Status    string `json:"status"`
		Additions int    `json:"additions"`
		Deletions int    `json:"deletions"`
		Changes   int    `json:"changes"`
	}

	fileSummaries := make([]fileSummary, 0, len(files))
	for _, f := range files {
		fileSummaries = append(fileSummaries, fileSummary{
			Path:      f.Filename,
			Status:    f.Status,
			Additions: f.Additions,
			Deletions: f.Deletions,
			Changes:   f.Changes,
		})
	}

	resp := struct {
		Repository  string        `json:"repository"`
		PRNumber    int           `json:"pr_number"`
		Title       string        `json:"title"`
		Body        string        `json:"body"`
		URL         string        `json:"url"`
		Author      string        `json:"author"`
		Draft       bool          `json:"draft"`
		BaseRef     string        `json:"base_ref"`
		HeadRef     string        `json:"head_ref"`
		BaseSHA     string        `json:"base_sha"`
		HeadSHA     string        `json:"head_sha"`
		Changed     int           `json:"changed_files"`
		Additions   int           `json:"additions"`
		Deletions   int           `json:"deletions"`
		CommitCount int           `json:"commit_count"`
		Commits     []any         `json:"commits"`
		Files       []fileSummary `json:"files"`
	}{
		Repository:  t.state.gh.Repo(),
		PRNumber:    prView.Number,
		Title:       prView.Title,
		Body:        prView.Body,
		URL:         prView.URL,
		Author:      prView.Author.Login,
		Draft:       prView.IsDraft,
		BaseRef:     prView.BaseRefName,
		HeadRef:     prView.HeadRefName,
		BaseSHA:     base,
		HeadSHA:     head,
		Changed:     prView.ChangedFiles,
		Additions:   prView.Additions,
		Deletions:   prView.Deletions,
		CommitCount: len(prView.Commits),
		Files:       fileSummaries,
	}

	for _, c := range prView.Commits {
		resp.Commits = append(resp.Commits, map[string]string{
			"sha":     c.OID,
			"message": c.Message,
		})
	}

	b, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encoding overview response: %w", err)
	}

	return textResult(string(b)), nil
}
