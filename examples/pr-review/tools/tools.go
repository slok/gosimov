package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
)

type GHClientConfig struct {
	Repo    string
	PR      int
	DryRun  bool
	Timeout time.Duration
}

type GHClient struct {
	repo    string
	pr      int
	dryRun  bool
	timeout time.Duration
}

func NewGHClient(cfg GHClientConfig) *GHClient {
	return &GHClient{
		repo:    cfg.Repo,
		pr:      cfg.PR,
		dryRun:  cfg.DryRun,
		timeout: cfg.Timeout,
	}
}

func (g *GHClient) Repo() string { return g.repo }

func (g *GHClient) PRNumber() int { return g.pr }

func (g *GHClient) DryRun() bool { return g.dryRun }

func (g *GHClient) run(ctx context.Context, args ...string) ([]byte, error) {
	tctx, cancel := context.WithTimeout(ctx, g.timeout)
	defer cancel()

	cmd := exec.CommandContext(tctx, "gh", args...)
	start := time.Now()
	log.Printf("[pr-review] gh exec start: %s", summarizeArgs(args))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		log.Printf("[pr-review] gh exec error: %s (duration=%s)", summarizeArgs(args), time.Since(start).Round(time.Millisecond))
		if tctx.Err() != nil {
			return nil, fmt.Errorf("gh command timed out: %w", tctx.Err())
		}

		return nil, fmt.Errorf("gh command failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	log.Printf("[pr-review] gh exec done: %s (duration=%s)", summarizeArgs(args), time.Since(start).Round(time.Millisecond))

	return stdout.Bytes(), nil
}

func (g *GHClient) runJSON(ctx context.Context, dst any, args ...string) error {
	b, err := g.run(ctx, args...)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(b, dst); err != nil {
		return fmt.Errorf("decoding gh JSON output: %w", err)
	}

	return nil
}

func (g *GHClient) PREndpoint(suffix string) string {
	base := fmt.Sprintf("repos/%s/pulls/%d", g.repo, g.pr)
	if suffix == "" {
		return base
	}

	return base + "/" + strings.TrimPrefix(suffix, "/")
}

type StateConfig struct {
	GH                *GHClient
	WorkDir           string
	ReviewMarker      string
	MaxInlineComments int
}

type State struct {
	gh           *GHClient
	workDir      string
	reviewMarker string

	maxInlineComments int

	mu                 sync.Mutex
	viewerLogin        string
	prHeadSHA          string
	prBaseSHA          string
	files              []prFile
	changedLinesByFile map[string]map[int]struct{}
	inlineComments     int
	summaryAction      string
}

func NewState(cfg StateConfig) *State {
	return &State{
		gh:                cfg.GH,
		workDir:           cfg.WorkDir,
		reviewMarker:      cfg.ReviewMarker,
		maxInlineComments: cfg.MaxInlineComments,
	}
}

func (s *State) WorkDir() string {
	return s.workDir
}

func (s *State) Warm(ctx context.Context) error {
	viewer, err := s.viewer(ctx)
	if err != nil {
		return err
	}

	if viewer == "" {
		return fmt.Errorf("empty GitHub viewer login")
	}

	if _, _, err := s.PRRefs(ctx); err != nil {
		return err
	}

	if _, err := s.PRFiles(ctx); err != nil {
		return err
	}

	return nil
}

func (s *State) viewer(ctx context.Context) (string, error) {
	s.mu.Lock()
	if s.viewerLogin != "" {
		v := s.viewerLogin
		s.mu.Unlock()
		return v, nil
	}
	s.mu.Unlock()

	if strings.EqualFold(strings.TrimSpace(os.Getenv("GITHUB_ACTIONS")), "true") {
		s.mu.Lock()
		s.viewerLogin = "github-actions[bot]"
		v := s.viewerLogin
		s.mu.Unlock()
		return v, nil
	}

	if actor := strings.TrimSpace(os.Getenv("GITHUB_ACTOR")); actor != "" {
		s.mu.Lock()
		s.viewerLogin = actor
		v := s.viewerLogin
		s.mu.Unlock()
		return v, nil
	}

	var data struct {
		Login string `json:"login"`
	}
	if err := s.gh.runJSON(ctx, &data, "api", "user"); err != nil {
		return "", fmt.Errorf("loading gh viewer: %w", err)
	}

	s.mu.Lock()
	s.viewerLogin = strings.TrimSpace(data.Login)
	v := s.viewerLogin
	s.mu.Unlock()

	return v, nil
}

func (s *State) PRRefs(ctx context.Context) (head string, base string, err error) {
	s.mu.Lock()
	if s.prHeadSHA != "" && s.prBaseSHA != "" {
		head = s.prHeadSHA
		base = s.prBaseSHA
		s.mu.Unlock()
		return head, base, nil
	}
	s.mu.Unlock()

	var pr struct {
		Head struct {
			SHA string `json:"sha"`
		} `json:"head"`
		Base struct {
			SHA string `json:"sha"`
		} `json:"base"`
	}

	if err := s.gh.runJSON(ctx, &pr, "api", s.gh.PREndpoint("")); err != nil {
		return "", "", fmt.Errorf("loading pr refs: %w", err)
	}

	head = strings.TrimSpace(pr.Head.SHA)
	base = strings.TrimSpace(pr.Base.SHA)
	if head == "" || base == "" {
		return "", "", fmt.Errorf("missing PR refs")
	}

	s.mu.Lock()
	s.prHeadSHA = head
	s.prBaseSHA = base
	s.mu.Unlock()

	return head, base, nil
}

type prFile struct {
	Filename  string `json:"filename"`
	Status    string `json:"status"`
	Additions int    `json:"additions"`
	Deletions int    `json:"deletions"`
	Changes   int    `json:"changes"`
	Patch     string `json:"patch"`
}

func (s *State) PRFiles(ctx context.Context) ([]prFile, error) {
	s.mu.Lock()
	if s.files != nil {
		files := make([]prFile, len(s.files))
		copy(files, s.files)
		s.mu.Unlock()
		return files, nil
	}
	s.mu.Unlock()

	all := make([]prFile, 0, 64)
	for page := 1; ; page++ {
		endpoint := fmt.Sprintf("%s/files?per_page=100&page=%d", s.gh.PREndpoint(""), page)
		var pageFiles []prFile
		if err := s.gh.runJSON(ctx, &pageFiles, "api", endpoint); err != nil {
			return nil, fmt.Errorf("loading PR files page %d: %w", page, err)
		}

		if len(pageFiles) == 0 {
			break
		}

		all = append(all, pageFiles...)
	}

	changed := make(map[string]map[int]struct{}, len(all))
	for _, f := range all {
		changed[f.Filename] = addedLinesFromPatch(f.Patch)
	}

	s.mu.Lock()
	s.files = make([]prFile, len(all))
	copy(s.files, all)
	s.changedLinesByFile = changed
	s.mu.Unlock()

	return all, nil
}

func (s *State) ValidateInlineTarget(path string, line int) error {
	if line <= 0 {
		return fmt.Errorf("line must be > 0")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.inlineComments >= s.maxInlineComments {
		return fmt.Errorf("inline comment budget exhausted (%d)", s.maxInlineComments)
	}

	lines, ok := s.changedLinesByFile[path]
	if !ok {
		return fmt.Errorf("path %q is not in changed files", path)
	}

	if len(lines) > 0 {
		if _, ok := lines[line]; !ok {
			return fmt.Errorf("line %d is not an added/changed right-side line in %q", line, path)
		}
	}

	return nil
}

func (s *State) incrementInlineComments() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inlineComments++
}

func (s *State) decrementInlineComments() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inlineComments > 0 {
		s.inlineComments--
	}
}

func (s *State) InlineCommentsPosted() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inlineComments
}

func (s *State) SetSummaryCommentAction(action string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.summaryAction = action
}

func (s *State) SummaryCommentAction() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.summaryAction
}

func textResult(text string) *tool.Result {
	return &tool.Result{
		Content: []model.ContentPart{{
			Type: model.ContentPartTypeText,
			Text: text,
		}},
	}
}

type issueComment struct {
	ID   int    `json:"id"`
	Body string `json:"body"`
	User struct {
		Login string `json:"login"`
	} `json:"user"`
}

func listIssueComments(ctx context.Context, gh *GHClient) ([]issueComment, error) {
	all := make([]issueComment, 0, 32)
	for page := 1; ; page++ {
		endpoint := fmt.Sprintf("repos/%s/issues/%d/comments?per_page=100&page=%d", gh.repo, gh.pr, page)
		var pageComments []issueComment
		if err := gh.runJSON(ctx, &pageComments, "api", endpoint); err != nil {
			return nil, fmt.Errorf("listing issue comments page %d: %w", page, err)
		}

		if len(pageComments) == 0 {
			break
		}
		all = append(all, pageComments...)
	}

	return all, nil
}

func intFromAny(v any) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case int64:
		return int(x)
	default:
		return 0
	}
}

var hunkHeaderRE = regexp.MustCompile(`^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@`)

func addedLinesFromPatch(patch string) map[int]struct{} {
	res := map[int]struct{}{}
	if strings.TrimSpace(patch) == "" {
		return res
	}

	lines := strings.Split(patch, "\n")
	newLine := 0
	inHunk := false

	for _, raw := range lines {
		if strings.HasPrefix(raw, "@@") {
			m := hunkHeaderRE.FindStringSubmatch(raw)
			if len(m) == 2 {
				v, err := strconv.Atoi(m[1])
				if err == nil {
					newLine = v
					inHunk = true
					continue
				}
			}
			inHunk = false
			continue
		}

		if !inHunk || raw == "" {
			continue
		}

		switch {
		case strings.HasPrefix(raw, "+++"):
			continue
		case strings.HasPrefix(raw, "+"):
			res[newLine] = struct{}{}
			newLine++
		case strings.HasPrefix(raw, "---"):
			continue
		case strings.HasPrefix(raw, "-"):
			continue
		case strings.HasPrefix(raw, " "):
			newLine++
		case strings.HasPrefix(raw, "\\"):
			continue
		default:
			newLine++
		}
	}

	return res
}

type loggingTool struct {
	inner tool.Tool
}

func WrapWithLogging(t tool.Tool) tool.Tool {
	if t == nil {
		return nil
	}

	return &loggingTool{inner: t}
}

func (t *loggingTool) ID() string {
	return t.inner.ID()
}

func (t *loggingTool) Description() string {
	return t.inner.Description()
}

func (t *loggingTool) Schema() json.RawMessage {
	return t.inner.Schema()
}

func (t *loggingTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	start := time.Now()
	log.Printf("[pr-review] tool start: %s", t.inner.ID())
	res, err := t.inner.Execute(ctx, args)
	if err != nil {
		log.Printf("[pr-review] tool error: %s (duration=%s): %s", t.inner.ID(), time.Since(start).Round(time.Millisecond), err)
		return nil, err
	}

	parts := 0
	if res != nil {
		parts = len(res.Content)
	}
	log.Printf("[pr-review] tool done: %s (duration=%s, content_parts=%d)", t.inner.ID(), time.Since(start).Round(time.Millisecond), parts)

	return res, nil
}

func summarizeArgs(args []string) string {
	if len(args) == 0 {
		return "gh"
	}

	max := len(args)
	if max > 6 {
		max = 6
	}

	parts := make([]string, 0, max)
	for i := 0; i < max; i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "body=") {
			parts = append(parts, "body=<redacted>")
			continue
		}
		parts = append(parts, arg)
	}

	if len(args) > max {
		parts = append(parts, fmt.Sprintf("...(%d args)", len(args)))
	}

	return "gh " + strings.Join(parts, " ")
}
