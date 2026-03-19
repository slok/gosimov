// Example viewer is a simple HTTP server that serves an HTML UI
// for browsing JSONL session files as formatted conversations.
//
// Each message kind (user, LLM, tool result) is styled differently.
// Tool call requests and metadata (tokens, model) are shown inline.
//
// Usage:
//
//	go run ./examples/viewer --dir /tmp
//	go run ./examples/viewer --dir /tmp --addr :9090
//	VIEWER_DIR=/tmp VIEWER_ADDR=:9090 go run ./examples/viewer
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/jsonl"
	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

type config struct {
	dir  string
	addr string
}

func loadConfig() (config, error) {
	var cfg config

	defaultAddr := os.Getenv("VIEWER_ADDR")
	if strings.TrimSpace(defaultAddr) == "" {
		defaultAddr = ":8080"
	}

	flag.StringVar(&cfg.dir, "dir", os.Getenv("VIEWER_DIR"), "Directory containing .jsonl files (or set VIEWER_DIR)")
	flag.StringVar(&cfg.addr, "addr", defaultAddr, "HTTP listen address (or set VIEWER_ADDR)")
	flag.Parse()

	cfg.dir = strings.TrimSpace(cfg.dir)
	cfg.addr = strings.TrimSpace(cfg.addr)

	if cfg.dir == "" {
		return config{}, fmt.Errorf("--dir is required (or set VIEWER_DIR)")
	}

	if cfg.addr == "" {
		return config{}, fmt.Errorf("--addr is required (or set VIEWER_ADDR)")
	}

	return cfg, nil
}

func run() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	repo, err := jsonl.New(jsonl.Config{Dir: cfg.dir})
	if err != nil {
		return fmt.Errorf("creating repository: %w", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", handleSessionList(repo))
	mux.HandleFunc("GET /sessions/{id}", handleSessionDetail(repo))
	mux.HandleFunc("GET /sessions/{id}/export", handleSessionExport(repo))

	fmt.Printf("Viewer: http://localhost%s\n", cfg.addr)
	fmt.Printf("Dir:    %s\n", cfg.dir)

	return http.ListenAndServe(cfg.addr, mux)
}

// --- Handlers ---

func handleSessionList(repo interface {
	store.SessionRepository
}) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		result, err := repo.ListSessions(r.Context(), store.ListOpts{})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := sessionListTmpl.Execute(w, result.Items); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func handleSessionDetail(repo interface {
	store.SessionRepository
	store.MessageRepository
}) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		ctx := r.Context()

		session, err := repo.GetSession(ctx, id)
		if err != nil {
			http.Error(w, fmt.Sprintf("session not found: %s", err), http.StatusNotFound)
			return
		}

		msgs, err := loadAllMessages(ctx, repo, id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		data := sessionDetailData{
			Session:    *session,
			Messages:   prepareMessages(msgs),
			Usage:      summarizeSessionUsage(msgs),
			IsExport:   false,
			ExportedAt: time.Now(),
		}

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := sessionDetailTmpl.Execute(w, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func handleSessionExport(repo interface {
	store.SessionRepository
	store.MessageRepository
}) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		ctx := r.Context()

		session, err := repo.GetSession(ctx, id)
		if err != nil {
			http.Error(w, fmt.Sprintf("session not found: %s", err), http.StatusNotFound)
			return
		}

		msgs, err := loadAllMessages(ctx, repo, id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		data := sessionDetailData{
			Session:    *session,
			Messages:   prepareMessages(msgs),
			Usage:      summarizeSessionUsage(msgs),
			IsExport:   true,
			ExportedAt: time.Now(),
		}

		filename := fmt.Sprintf("gosimov-session-%s.html", id)
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filename))
		if err := sessionDetailTmpl.Execute(w, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

// --- Data types for templates ---

type sessionDetailData struct {
	Session    model.Session
	Messages   []messageView
	Usage      sessionUsageSummary
	IsExport   bool
	ExportedAt time.Time
}

type sessionUsageSummary struct {
	Total      int
	Input      int
	Output     int
	CacheRead  int
	CacheWrite int
}

type messageView struct {
	ID         string
	Kind       string
	KindLabel  string
	CSSClass   string
	TextHTML   template.HTML
	Turn       int
	TurnStart  bool
	ToolCalls  []toolCallView
	ToolID     string
	IsError    bool
	Model      string
	Tokens     string
	Checkpoint string // Compaction: first kept message ID.
	TokensPre  string // Compaction: tokens before compaction.
	CreatedAt  time.Time
}

type toolCallView struct {
	ToolID    string
	Arguments string
}

var markdownRenderer = goldmark.New(
	goldmark.WithExtensions(extension.GFM),
)

// --- Helpers ---

// loadAllMessages fetches all messages for a session using pagination.
func loadAllMessages(ctx context.Context, repo store.MessageRepository, sessionID string) ([]model.Message, error) {
	var all []model.Message
	opts := store.ListOpts{Limit: 100}

	for {
		result, err := repo.ListMessages(ctx, sessionID, opts)
		if err != nil {
			return nil, err
		}

		all = append(all, result.Items...)

		if result.NextCursor == "" {
			break
		}
		opts.Cursor = result.NextCursor
	}

	return all, nil
}

// prepareMessages converts model messages to template-friendly views.
func prepareMessages(msgs []model.Message) []messageView {
	views := make([]messageView, 0, len(msgs))
	turnByMsgID, turnStartByMsgID := buildTurnIndex(msgs)

	for _, msg := range msgs {
		v := messageView{
			ID:        msg.ID,
			Kind:      string(msg.Kind),
			CreatedAt: msg.CreatedAt,
			Turn:      turnByMsgID[msg.ID],
			TurnStart: turnStartByMsgID[msg.ID],
		}

		switch msg.Kind {
		case model.MessageKindUser:
			v.KindLabel = "User"
			v.CSSClass = "msg-user"
		case model.MessageKindLLM:
			v.KindLabel = "LLM"
			v.CSSClass = "msg-llm"
		case model.MessageKindToolResult:
			v.KindLabel = "Tool Result"
			v.CSSClass = "msg-tool"
			if msg.IsError {
				v.CSSClass = "msg-tool-error"
				v.IsError = true
			}
			v.ToolID = msg.ToolCallID
		case model.MessageKindCompaction:
			v.KindLabel = "Compaction"
			v.CSSClass = "msg-compaction"
			if msg.Compaction != nil {
				v.Checkpoint = msg.Compaction.FirstKeptID
				if msg.Compaction.TokensBefore > 0 {
					v.TokensPre = fmt.Sprintf("%d tokens before", msg.Compaction.TokensBefore)
				}
			}
		}

		// Text content.
		var parts []string
		for _, cp := range msg.Content {
			switch cp.Type {
			case model.ContentPartTypeText:
				parts = append(parts, cp.Text)
			case model.ContentPartTypeImage:
				parts = append(parts, "[image]")
			}
		}
		v.TextHTML = renderMarkdown(strings.Join(parts, "\n"))

		// Tool call requests (LLM messages).
		for _, tc := range msg.ToolCallRequests {
			args := string(tc.Arguments)
			// Pretty-print JSON arguments.
			var parsed any
			if json.Unmarshal(tc.Arguments, &parsed) == nil {
				if pretty, err := json.MarshalIndent(parsed, "", "  "); err == nil {
					args = string(pretty)
				}
			}
			v.ToolCalls = append(v.ToolCalls, toolCallView{
				ToolID:    tc.ToolID,
				Arguments: args,
			})
		}

		// Metadata (LLM messages).
		if msg.Metadata != nil {
			v.Model = msg.Metadata.Model
			if msg.Metadata.Usage != nil {
				u := msg.Metadata.Usage
				v.Tokens = fmt.Sprintf("%d total (%d in / %d out)", u.TotalTokens, u.InputTokens, u.OutputTokens)
				if u.CacheReadTokens > 0 || u.CacheWriteTokens > 0 {
					v.Tokens += fmt.Sprintf(" cache r/w %d/%d", u.CacheReadTokens, u.CacheWriteTokens)
				}
			}
		}

		views = append(views, v)
	}

	return views
}

func summarizeSessionUsage(msgs []model.Message) sessionUsageSummary {
	s := sessionUsageSummary{}

	for _, msg := range msgs {
		if msg.Kind != model.MessageKindLLM || msg.Metadata == nil || msg.Metadata.Usage == nil {
			continue
		}

		u := msg.Metadata.Usage
		s.Total += u.TotalTokens
		s.Input += u.InputTokens
		s.Output += u.OutputTokens
		s.CacheRead += u.CacheReadTokens
		s.CacheWrite += u.CacheWriteTokens
	}

	return s
}

func buildTurnIndex(messages []model.Message) (map[string]int, map[string]bool) {
	turnByMsgID := map[string]int{}
	turnStartByMsgID := map[string]bool{}

	turns := model.TurnsFromMessages(messages)
	for i, turn := range turns {
		turnNumber := i + 1
		for j, msg := range turn.Messages {
			if msg.ID == "" {
				continue
			}

			turnByMsgID[msg.ID] = turnNumber
			if j == 0 {
				turnStartByMsgID[msg.ID] = true
			}
		}
	}

	return turnByMsgID, turnStartByMsgID
}

func renderMarkdown(text string) template.HTML {
	if strings.TrimSpace(text) == "" {
		return ""
	}

	var out strings.Builder
	if err := markdownRenderer.Convert([]byte(text), &out); err != nil {
		return template.HTML("<pre>" + template.HTMLEscapeString(text) + "</pre>")
	}

	return template.HTML(out.String())
}

// --- Templates ---

var funcs = template.FuncMap{
	"formatTime": func(t time.Time) string {
		if t.IsZero() {
			return ""
		}
		return t.Format("2006-01-02 15:04:05")
	},
}

var sessionListTmpl = template.Must(template.New("list").Funcs(funcs).Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gosimov Sessions</title>
<style>` + cssStyles + `</style>
</head>
<body>
<div class="container">
<h1>Sessions</h1>
{{if not .}}
<p class="empty">No sessions found.</p>
{{else}}
<div class="session-list">
{{range .}}
<a href="/sessions/{{.ID}}" class="session-card">
  <span class="session-id">{{.ID}}</span>
  <span class="session-time">{{formatTime .CreatedAt}}</span>
</a>
{{end}}
</div>
{{end}}
</div>
</body>
</html>
`))

var sessionDetailTmpl = template.Must(template.New("detail").Funcs(funcs).Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session {{.Session.ID}}</title>
<style>` + cssStyles + `</style>
</head>
<body>
<div class="container">
<div class="header">
  {{if not .IsExport}}
  <div class="header-actions">
  <a href="/" class="back">&larr; Sessions</a>
  <a href="/sessions/{{.Session.ID}}/export" class="export">Export HTML</a>
  </div>
  {{end}}
  <h1>Session <code>{{.Session.ID}}</code></h1>
  <p class="meta">Created: {{formatTime .Session.CreatedAt}} &middot; {{len .Messages}} messages{{if .IsExport}} &middot; Exported: {{formatTime .ExportedAt}}{{end}}</p>
  <p class="meta">Usage: {{.Usage.Total}} total &middot; {{.Usage.Input}} in &middot; {{.Usage.Output}} out &middot; cache r/w {{.Usage.CacheRead}}/{{.Usage.CacheWrite}}</p>
</div>

<div class="conversation">
{{range .Messages}}
{{if .TurnStart}}
<div class="turn-divider">Turn {{.Turn}}</div>
{{end}}
<div class="msg {{.CSSClass}}">
  <div class="msg-header">
    <span class="msg-kind">{{.KindLabel}}</span>
    {{if .Turn}}<span class="msg-turn">turn {{.Turn}}</span>{{end}}
    {{if .Model}}<span class="msg-model">{{.Model}}</span>{{end}}
    {{if .Tokens}}<span class="msg-tokens">{{.Tokens}}</span>{{end}}
    {{if .ToolID}}<span class="msg-tool-id">{{.ToolID}}</span>{{end}}
    {{if .IsError}}<span class="msg-error-badge">ERROR</span>{{end}}
    {{if .Checkpoint}}<span class="msg-checkpoint">checkpoint: {{.Checkpoint}}</span>{{end}}
    {{if .TokensPre}}<span class="msg-tokens-pre">{{.TokensPre}}</span>{{end}}
    <span class="msg-time">{{formatTime .CreatedAt}}</span>
  </div>

  {{if .ToolCalls}}
  <div class="tool-calls">
    {{range .ToolCalls}}
    <div class="tool-call">
      <div class="tool-call-header">Tool call: <strong>{{.ToolID}}</strong></div>
      <pre class="tool-call-args">{{.Arguments}}</pre>
    </div>
    {{end}}
  </div>
  {{end}}

	{{if .TextHTML}}
	<div class="msg-text markdown-body">{{.TextHTML}}</div>
	{{end}}
</div>
{{end}}
</div>
</div>
</body>
</html>
`))

const cssStyles = `
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #f6f8fb;
  color: #1f2937;
  line-height: 1.5;
}
.container { max-width: 900px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 1.5rem; margin-bottom: 16px; color: #111827; }
h1 code { font-size: 0.85em; background: #eef2ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
.header-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}
.back { display: inline-block; font-size: 0.9rem; }
.export {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #1d4ed8;
  color: #fff;
  border-radius: 8px;
  padding: 6px 12px;
  font-size: 0.82rem;
  font-weight: 600;
  border: 1px solid #1d4ed8;
}
.export:hover {
  text-decoration: none;
  background: #1e40af;
  border-color: #1e40af;
}
.meta { font-size: 0.85rem; color: #6b7280; margin-bottom: 24px; }
.empty { color: #6b7280; font-style: italic; }

/* Session list */
.session-list { display: flex; flex-direction: column; gap: 8px; }
.session-card {
  display: flex; justify-content: space-between; align-items: center;
  background: #ffffff; border: 1px solid #dbe1ea; border-radius: 8px;
  padding: 14px 18px; transition: border-color 0.15s;
}
.session-card:hover { border-color: #60a5fa; text-decoration: none; }
.session-id { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9rem; color: #111827; }
.session-time { font-size: 0.8rem; color: #6b7280; }

/* Conversation */
.conversation { display: flex; flex-direction: column; gap: 12px; }
.turn-divider {
  margin: 10px 0 2px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  color: #475569;
  border-top: 1px solid #cbd5e1;
  padding-top: 8px;
}
.msg {
  border-radius: 10px; padding: 12px 16px;
  border-left: 4px solid transparent;
  border: 1px solid #e5e7eb;
  max-width: 95%;
}
.msg-header {
  display: flex; align-items: center; gap: 10px;
  font-size: 0.75rem; margin-bottom: 8px; flex-wrap: wrap;
}
.msg-kind {
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
}
.msg-model, .msg-tokens, .msg-tool-id {
  color: #6b7280; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.75rem;
}
.msg-turn {
  color: #334155;
  background: #e2e8f0;
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
}
.msg-time { color: #9ca3af; margin-left: auto; font-size: 0.7rem; }
.msg-error-badge {
  background: #dc2626; color: #fff; font-size: 0.65rem; font-weight: 700;
  padding: 1px 6px; border-radius: 3px; letter-spacing: 0.5px;
}

/* User messages */
.msg-user {
  background: #eff6ff; border-left-color: #3b82f6;
  align-self: flex-end;
}
.msg-user .msg-kind { color: #1d4ed8; }

/* LLM messages */
.msg-llm {
  background: #ffffff; border-left-color: #94a3b8;
  align-self: flex-start;
}
.msg-llm .msg-kind { color: #334155; }

/* Tool result */
.msg-tool {
  background: #ecfdf5; border-left-color: #16a34a;
  align-self: flex-start; font-size: 0.9rem;
}
.msg-tool .msg-kind { color: #15803d; }

.msg-tool-error {
  background: #fef2f2; border-left-color: #dc2626;
  align-self: flex-start; font-size: 0.9rem;
}
.msg-tool-error .msg-kind { color: #dc2626; }

/* Compaction messages */
.msg-compaction {
  background: #fff7ed; border-left-color: #ea580c;
  align-self: stretch; font-size: 0.9rem;
}
.msg-compaction .msg-kind { color: #c2410c; }
.msg-checkpoint, .msg-tokens-pre {
  color: #6b7280; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.75rem;
}

/* Tool calls */
.tool-calls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 8px; }
.tool-call {
  background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px;
  overflow: hidden;
}
.tool-call-header {
  padding: 6px 12px; background: #f3f4f6; font-size: 0.8rem; color: #6b7280;
  border-bottom: 1px solid #d1d5db;
}
.tool-call-header strong { color: #4f46e5; }
.tool-call-args {
  padding: 10px 12px; font-size: 0.8rem; color: #111827;
  white-space: pre-wrap; word-wrap: break-word; margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Markdown rendering */
.markdown-body {
  font-size: 0.92rem;
  line-height: 1.65;
  color: #111827;
}
.markdown-body > :first-child { margin-top: 0; }
.markdown-body > :last-child { margin-bottom: 0; }
.markdown-body p,
.markdown-body ul,
.markdown-body ol,
.markdown-body blockquote,
.markdown-body pre,
.markdown-body table,
.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4 {
  margin: 0 0 10px 0;
}
.markdown-body ul,
.markdown-body ol {
  padding-left: 20px;
}
.markdown-body code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  background: #eef2ff;
  color: #3730a3;
  border-radius: 4px;
  padding: 1px 4px;
  font-size: 0.86em;
}
.markdown-body pre {
  background: #f3f4f6;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  padding: 10px 12px;
  overflow-x: auto;
}
.markdown-body pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}
.markdown-body blockquote {
  border-left: 3px solid #cbd5e1;
  color: #475569;
  padding-left: 10px;
}
`
