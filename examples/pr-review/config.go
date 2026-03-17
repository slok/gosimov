package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

var trustedAssociations = map[string]struct{}{
	"OWNER":        {},
	"MEMBER":       {},
	"COLLABORATOR": {},
}

type config struct {
	apiKey            string
	modelID           string
	repo              string
	prNumber          int
	eventName         string
	eventPath         string
	mention           string
	workDir           string
	maxIterations     int
	maxInlineComments int
	ghTimeout         time.Duration
	dryRun            bool
}

func loadConfig() (config, error) {
	var cfg config

	flag.StringVar(&cfg.apiKey, "api-key", envFirst("OPENCODE_GO_API_KEY", "INTEGRATION_OPENCODE_GO_API_KEY"), "OpenCode Go API key")
	flag.StringVar(&cfg.modelID, "model", envFirst("OPENCODE_GO_MODEL", "INTEGRATION_OPENCODE_GO_MODEL", "minimax-m2.5"), "OpenCode Go model ID")
	flag.StringVar(&cfg.repo, "repo", strings.TrimSpace(os.Getenv("GITHUB_REPOSITORY")), "GitHub repository owner/name")
	flag.IntVar(&cfg.prNumber, "pr", 0, "Pull request number")
	flag.StringVar(&cfg.eventName, "event-name", strings.TrimSpace(os.Getenv("GITHUB_EVENT_NAME")), "GitHub event name")
	flag.StringVar(&cfg.eventPath, "event-path", strings.TrimSpace(os.Getenv("GITHUB_EVENT_PATH")), "Path to GitHub event payload JSON")
	flag.StringVar(&cfg.mention, "mention", "@gosimov-review", "Trigger mention required in PR body/title or comment")
	flag.StringVar(&cfg.workDir, "work-dir", defaultString(os.Getenv("GITHUB_WORKSPACE"), "."), "Directory used by local repo read/list tools")
	flag.IntVar(&cfg.maxIterations, "max-iterations", 32, "Maximum LLM iterations for the review turn")
	flag.IntVar(&cfg.maxInlineComments, "max-inline-comments", 12, "Maximum inline comments tool can post")
	flag.DurationVar(&cfg.ghTimeout, "gh-timeout", 30*time.Second, "Timeout per gh command")
	flag.BoolVar(&cfg.dryRun, "dry-run", false, "Do not post comments/reviews; print intended actions")

	flag.Parse()

	cfg.apiKey = strings.TrimSpace(cfg.apiKey)
	cfg.modelID = strings.TrimSpace(cfg.modelID)
	cfg.repo = strings.TrimSpace(cfg.repo)
	cfg.eventName = strings.TrimSpace(cfg.eventName)
	cfg.eventPath = strings.TrimSpace(cfg.eventPath)
	cfg.mention = strings.TrimSpace(cfg.mention)
	cfg.workDir = strings.TrimSpace(cfg.workDir)

	if cfg.apiKey == "" {
		return config{}, fmt.Errorf("--api-key is required (or set OPENCODE_GO_API_KEY)")
	}
	if cfg.modelID == "" {
		return config{}, fmt.Errorf("--model is required")
	}
	if cfg.repo == "" {
		return config{}, fmt.Errorf("--repo is required (or set GITHUB_REPOSITORY)")
	}
	if cfg.maxIterations <= 0 {
		return config{}, fmt.Errorf("--max-iterations must be > 0")
	}
	if cfg.maxInlineComments <= 0 {
		return config{}, fmt.Errorf("--max-inline-comments must be > 0")
	}
	if cfg.ghTimeout <= 0 {
		return config{}, fmt.Errorf("--gh-timeout must be > 0")
	}

	return cfg, nil
}

type reviewContext struct {
	Repo              string
	PRNumber          int
	Mention           string
	EventName         string
	ActorAssociation  string
	TriggeredBy       string
	ShouldRun         bool
	SkipReason        string
	TriggerSourceText string
}

func resolveReviewContext(cfg config) (reviewContext, error) {
	rctx := reviewContext{
		Repo:      cfg.repo,
		PRNumber:  cfg.prNumber,
		Mention:   cfg.mention,
		EventName: cfg.eventName,
	}

	if cfg.eventPath == "" {
		if rctx.PRNumber <= 0 {
			return reviewContext{}, fmt.Errorf("--pr is required when no --event-path is provided")
		}
		rctx.ShouldRun = true
		rctx.TriggeredBy = "manual"
		return rctx, nil
	}

	b, err := os.ReadFile(cfg.eventPath)
	if err != nil {
		return reviewContext{}, fmt.Errorf("reading event payload: %w", err)
	}

	var payload githubEventPayload
	if err := json.Unmarshal(b, &payload); err != nil {
		return reviewContext{}, fmt.Errorf("decoding event payload: %w", err)
	}

	if payload.Repository.FullName != "" {
		rctx.Repo = payload.Repository.FullName
	}

	switch cfg.eventName {
	case "issue_comment":
		if payload.Issue.PullRequest == nil {
			rctx.SkipReason = "event is issue comment but not on a pull request"
			return rctx, nil
		}
		if payload.Issue.Number > 0 {
			rctx.PRNumber = payload.Issue.Number
		}
		rctx.ActorAssociation = strings.ToUpper(strings.TrimSpace(payload.Comment.AuthorAssociation))
		rctx.TriggerSourceText = payload.Comment.Body
		rctx.TriggeredBy = "issue_comment"

	case "pull_request_target":
		if payload.PullRequest.Number > 0 {
			rctx.PRNumber = payload.PullRequest.Number
		}
		rctx.ActorAssociation = strings.ToUpper(strings.TrimSpace(payload.PullRequest.AuthorAssociation))
		rctx.TriggerSourceText = payload.PullRequest.Title + "\n" + payload.PullRequest.Body
		rctx.TriggeredBy = "pull_request_target"

	default:
		if rctx.PRNumber <= 0 {
			return reviewContext{}, fmt.Errorf("unsupported --event-name %q and --pr not set", cfg.eventName)
		}
		rctx.ShouldRun = true
		rctx.TriggeredBy = "manual"
		return rctx, nil
	}

	if rctx.PRNumber <= 0 {
		return reviewContext{}, fmt.Errorf("could not determine PR number from event payload")
	}

	if _, ok := trustedAssociations[rctx.ActorAssociation]; !ok {
		rctx.SkipReason = fmt.Sprintf("author association %q is not trusted", rctx.ActorAssociation)
		return rctx, nil
	}

	if !strings.Contains(strings.ToLower(rctx.TriggerSourceText), strings.ToLower(cfg.mention)) {
		rctx.SkipReason = fmt.Sprintf("mention %q not present in trigger text", cfg.mention)
		return rctx, nil
	}

	rctx.ShouldRun = true
	return rctx, nil
}

type githubEventPayload struct {
	Repository struct {
		FullName string `json:"full_name"`
	} `json:"repository"`
	Issue struct {
		Number      int `json:"number"`
		PullRequest *struct {
			URL string `json:"url"`
		} `json:"pull_request"`
	} `json:"issue"`
	Comment struct {
		Body              string `json:"body"`
		AuthorAssociation string `json:"author_association"`
	} `json:"comment"`
	PullRequest struct {
		Number            int    `json:"number"`
		Title             string `json:"title"`
		Body              string `json:"body"`
		AuthorAssociation string `json:"author_association"`
	} `json:"pull_request"`
}

func buildUserPrompt(rctx reviewContext, cfg config) string {
	mode := "live"
	if cfg.dryRun {
		mode = "dry-run"
	}

	return strings.TrimSpace(strings.Join([]string{
		fmt.Sprintf("Review pull request #%d on repository %s.", rctx.PRNumber, rctx.Repo),
		fmt.Sprintf("Trigger source: %s. Mode: %s.", rctx.TriggeredBy, mode),
		"Use the GitHub tools to inspect overview, diffs, discussions, and files.",
		"You may use local ls/read tools for extra repository context.",
		"When done, publish exactly one markdown summary using pr_comment_upsert.",
		"Post inline comments only for actionable issues that clearly need changes.",
		"Avoid duplicate or low-value nit comments.",
	}, "\n"))
}

func buildSystemPrompt(marker string, maxInline int) string {
	return strings.TrimSpace(strings.Join([]string{
		"You are an expert pull request reviewer focused on correctness, reliability, maintainability, and tests.",
		"Be practical: do not be a style zealot, avoid trivial nits, and do not invent issues.",
		"Always inspect context before commenting: read overview first, then relevant diffs/files/discussion.",
		"When posting a summary with pr_comment_upsert, include this marker once: " + marker,
		"Summary format in markdown:",
		"- Title: Gosimov Automated Review",
		"- Short verdict",
		"- Key risks and why they matter",
		"- Concrete follow-ups",
		"For inline comments, keep them concise, specific, and tied to the exact changed line.",
		"Do not exceed the inline comment budget. Current max: " + strconv.Itoa(maxInline) + ".",
		"If no actionable issues are found, post a brief LGTM-style summary and no inline comments.",
	}, "\n"))
}

func envFirst(keys ...string) string {
	for _, key := range keys {
		if v := strings.TrimSpace(os.Getenv(key)); v != "" {
			return v
		}
	}

	return ""
}

func defaultString(v string, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}

	return v
}
