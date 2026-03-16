package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/subscriber"
)

func (a *app) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", a.handleIndex)
	mux.HandleFunc("GET /api/sessions", a.handleListSessions)
	mux.HandleFunc("POST /api/sessions", a.handleCreateSession)
	mux.HandleFunc("POST /api/sessions/load", a.handleLoadSession)
	mux.HandleFunc("POST /api/prompt", a.handlePromptAny)
	mux.HandleFunc("POST /api/compact", a.handleCompactAny)
	mux.HandleFunc("POST /api/stop", a.handleStopAny)
	mux.HandleFunc("GET /api/sessions/{id}/status", a.handleSessionStatus)
	mux.HandleFunc("GET /api/sessions/{id}/context-stats", a.handleContextStats)
	mux.HandleFunc("GET /api/sessions/{id}/export", a.handleExportSession)
	mux.HandleFunc("POST /api/sessions/{id}/prompt", a.handlePrompt)
	mux.HandleFunc("GET /api/sessions/{id}/events", a.handleSessionEvents)

	return mux
}

func (a *app) handleIndex(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := htmlTemplates.ExecuteTemplate(w, "index.html", nil); err != nil {
		http.Error(w, fmt.Sprintf("rendering index: %s", err), http.StatusInternalServerError)
	}
}

func (a *app) handleCreateSession(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	log.Printf("create-session start remote=%s", r.RemoteAddr)
	defer func() {
		log.Printf("create-session end duration=%s", time.Since(started))
	}()

	ctx := r.Context()
	workDir, err := os.MkdirTemp(a.cfg.workDir, "session-*")
	if err != nil {
		log.Printf("create-session mkdirtemp error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	tools, err := createToolsForDir(workDir)
	if err != nil {
		log.Printf("create-session tools error work_dir=%s err=%v", workDir, err)
		http.Error(w, fmt.Sprintf("creating session tools: %s", err), http.StatusInternalServerError)
		return
	}

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          a.provider,
		Compactor:         a.compactor,
		SystemPrompt:      defaultSystemPrompt(a.cfg.systemPrompt),
		Tools:             tools,
		SessionRepository: a.sessRepo,
		MessageRepository: a.msgRepo,
		TurnMaxIterations: a.cfg.maxIterations,
		Logger:            a.sdkLogger,
	})
	if err != nil {
		log.Printf("create-session new-session error: %v", err)
		http.Error(w, fmt.Sprintf("creating session: %s", err), http.StatusInternalServerError)
		return
	}

	a.mu.Lock()
	a.sessions[session.Session().ID] = &chatSession{session: session, workDir: workDir}
	a.mu.Unlock()

	log.Printf("create-session ok session_id=%s work_dir=%s", session.Session().ID, workDir)

	writeJSON(w, http.StatusCreated, map[string]string{"id": session.Session().ID, "work_dir": workDir})
}

func (a *app) handleListSessions(w http.ResponseWriter, r *http.Request) {
	result, err := a.sessRepo.ListSessions(r.Context(), store.ListOpts{Limit: 200})
	if err != nil {
		http.Error(w, fmt.Sprintf("listing sessions: %s", err), http.StatusInternalServerError)
		return
	}

	type sessionItem struct {
		ID        string    `json:"id"`
		CreatedAt time.Time `json:"created_at"`
	}

	items := make([]sessionItem, 0, len(result.Items))
	for _, s := range result.Items {
		items = append(items, sessionItem{ID: s.ID, CreatedAt: s.CreatedAt})
	}

	writeJSON(w, http.StatusOK, map[string]any{"sessions": items})
}

func (a *app) handleLoadSession(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	log.Printf("load-session start remote=%s", r.RemoteAddr)
	defer func() {
		log.Printf("load-session end duration=%s", time.Since(started))
	}()

	req, err := parsePromptRequest(r)
	if err != nil {
		log.Printf("load-session parse error: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	sessionID := strings.TrimSpace(req.SessionID)
	if sessionID == "" {
		log.Printf("load-session missing session id")
		http.Error(w, "session_id is required", http.StatusBadRequest)
		return
	}

	if existing := a.getSession(sessionID); existing != nil {
		log.Printf("load-session cached session_id=%s", sessionID)
		writeJSON(w, http.StatusOK, map[string]string{"id": sessionID})
		return
	}

	workDir, err := os.MkdirTemp(a.cfg.workDir, "session-*")
	if err != nil {
		log.Printf("load-session mkdirtemp error session_id=%s err=%v", sessionID, err)
		http.Error(w, fmt.Sprintf("creating session work dir: %s", err), http.StatusInternalServerError)
		return
	}

	tools, err := createToolsForDir(workDir)
	if err != nil {
		log.Printf("load-session tools error session_id=%s work_dir=%s err=%v", sessionID, workDir, err)
		http.Error(w, fmt.Sprintf("creating session tools: %s", err), http.StatusInternalServerError)
		return
	}

	session, err := agent.LoadSession(r.Context(), agent.LoadSessionConfig{
		SessionID:         sessionID,
		Provider:          a.provider,
		Compactor:         a.compactor,
		SystemPrompt:      defaultSystemPrompt(a.cfg.systemPrompt),
		Tools:             tools,
		SessionRepository: a.sessRepo,
		TurnMaxIterations: a.cfg.maxIterations,
		Logger:            a.sdkLogger,
	})
	if err != nil {
		log.Printf("load-session load error session_id=%s err=%v", sessionID, err)
		http.Error(w, fmt.Sprintf("loading session: %s", err), http.StatusInternalServerError)
		return
	}

	loaded, err := loadAllMessages(r.Context(), a.msgRepo, sessionID)
	if err != nil {
		log.Printf("load-session messages error session_id=%s err=%v", sessionID, err)
		http.Error(w, fmt.Sprintf("loading session messages: %s", err), http.StatusInternalServerError)
		return
	}

	loadedMsgs := len(loaded)
	// Always run loaded history through trim/sanitize. Even when max-history does
	// not cut anything, sanitization removes incomplete tool-use tails that would
	// be rejected by strict providers on the next continue.
	trimmed := trimLoadedMessages(loaded, a.cfg.maxHistory)
	if len(trimmed) != loadedMsgs {
		log.Printf("load-session trimmed history session_id=%s from=%d to=%d", sessionID, loadedMsgs, len(trimmed))
		loadedMsgs = len(trimmed)
	}

	for _, msg := range trimmed {
		session.AppendMessage(msg)
	}

	log.Printf("load-session ok session_id=%s loaded_messages=%d work_dir=%s", sessionID, loadedMsgs, workDir)

	a.mu.Lock()
	a.sessions[sessionID] = &chatSession{session: session, workDir: workDir}
	a.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]string{"id": sessionID})
}

func (a *app) handlePrompt(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	started := time.Now()
	log.Printf("prompt start session_id=%s remote=%s", id, r.RemoteAddr)
	defer func() {
		log.Printf("prompt end session_id=%s duration=%s", id, time.Since(started))
	}()

	cSession := a.getSession(id)
	if cSession == nil {
		log.Printf("prompt missing-session session_id=%s", id)
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	reqText, err := parsePromptText(r)
	if err != nil {
		log.Printf("prompt parse error session_id=%s err=%v", id, err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	log.Printf("prompt context session_id=%s messages=%d", id, len(cSession.session.Messages()))

	reqText = strings.TrimSpace(reqText)
	if reqText == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}

	promptCtx := context.WithValue(r.Context(), contextKeySessionID{}, id)
	promptCtx, finishOp := cSession.startOperation(promptCtx)
	defer finishOp()
	result, err := cSession.session.Prompt(promptCtx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: reqText}}, agent.PromptOptions{})
	if err != nil {
		log.Printf("prompt execution error session_id=%s err=%v", id, err)
		http.Error(w, fmt.Sprintf("prompt failed: %s", err), http.StatusInternalServerError)
		return
	}

	if shouldRetryForExecutionEvidence(result) {
		log.Printf("prompt unverified-claim retry session_id=%s", id)
		retryText := "Execution policy reminder: you claimed changes were made, but no tool output was produced in the previous step. Now execute the required tools and provide evidence from tool results. Do not claim changes without tool evidence."
		if _, retryErr := cSession.session.Prompt(promptCtx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: retryText}}, agent.PromptOptions{}); retryErr != nil {
			log.Printf("prompt retry error session_id=%s err=%v", id, retryErr)
		}
	}

	log.Printf("prompt ok session_id=%s", id)

	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (a *app) handlePromptAny(w http.ResponseWriter, r *http.Request) {
	req, err := parsePromptRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if strings.TrimSpace(req.SessionID) == "" {
		http.Error(w, "session_id is required", http.StatusBadRequest)
		return
	}

	r.SetPathValue("id", req.SessionID)

	if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		a.handlePrompt(w, r)
		return
	}

	id := req.SessionID
	started := time.Now()
	log.Printf("prompt start session_id=%s remote=%s", id, r.RemoteAddr)
	defer func() {
		log.Printf("prompt end session_id=%s duration=%s", id, time.Since(started))
	}()

	cSession := a.getSession(id)
	if cSession == nil {
		log.Printf("prompt missing-session session_id=%s", id)
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	log.Printf("prompt context session_id=%s messages=%d", id, len(cSession.session.Messages()))

	req.Text = strings.TrimSpace(req.Text)
	if req.Text == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}

	promptCtx := context.WithValue(r.Context(), contextKeySessionID{}, id)
	promptCtx, finishOp := cSession.startOperation(promptCtx)
	defer finishOp()
	result, err := cSession.session.Prompt(promptCtx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: req.Text}}, agent.PromptOptions{})
	if err != nil {
		log.Printf("prompt execution error session_id=%s err=%v", id, err)
		http.Error(w, fmt.Sprintf("prompt failed: %s", err), http.StatusInternalServerError)
		return
	}

	if shouldRetryForExecutionEvidence(result) {
		log.Printf("prompt unverified-claim retry session_id=%s", id)
		retryText := "Execution policy reminder: you claimed changes were made, but no tool output was produced in the previous step. Now execute the required tools and provide evidence from tool results. Do not claim changes without tool evidence."
		if _, retryErr := cSession.session.Prompt(promptCtx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: retryText}}, agent.PromptOptions{}); retryErr != nil {
			log.Printf("prompt retry error session_id=%s err=%v", id, retryErr)
		}
	}

	log.Printf("prompt ok session_id=%s", id)
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (a *app) handleCompactAny(w http.ResponseWriter, r *http.Request) {
	req, err := parsePromptRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	sessionID := strings.TrimSpace(req.SessionID)
	if sessionID == "" {
		http.Error(w, "session_id is required", http.StatusBadRequest)
		return
	}

	started := time.Now()
	log.Printf("compact start session_id=%s remote=%s", sessionID, r.RemoteAddr)
	defer func() {
		log.Printf("compact end session_id=%s duration=%s", sessionID, time.Since(started))
	}()

	cSession := a.getSession(sessionID)
	if cSession == nil {
		log.Printf("compact missing-session session_id=%s", sessionID)
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	compactCtx := context.WithValue(r.Context(), contextKeySessionID{}, sessionID)
	compactCtx, finishOp := cSession.startOperation(compactCtx)
	defer finishOp()
	result, err := cSession.session.Compact(compactCtx)
	if err != nil {
		log.Printf("compact error session_id=%s err=%v", sessionID, err)
		http.Error(w, fmt.Sprintf("compact failed: %s", err), http.StatusInternalServerError)
		return
	}

	compacted := result != nil && result.Message != nil

	log.Printf("compact ok session_id=%s compacted=%t", sessionID, compacted)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "compacted": compacted})
}

func (a *app) handleStopAny(w http.ResponseWriter, r *http.Request) {
	req, err := parsePromptRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	sessionID := strings.TrimSpace(req.SessionID)
	if sessionID == "" {
		http.Error(w, "session_id is required", http.StatusBadRequest)
		return
	}

	cSession := a.getSession(sessionID)
	if cSession == nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	op, stopped := cSession.stopOperation()
	if stopped {
		log.Printf("stop ok session_id=%s operation=%s", sessionID, op)
	} else {
		log.Printf("stop noop session_id=%s", sessionID)
	}

	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "stopped": stopped, "operation": op})
}

func (a *app) handleExportSession(w http.ResponseWriter, r *http.Request) {
	sessionID := strings.TrimSpace(r.PathValue("id"))
	started := time.Now()
	log.Printf("export start session_id=%s remote=%s", sessionID, r.RemoteAddr)
	defer func() {
		log.Printf("export end session_id=%s duration=%s", sessionID, time.Since(started))
	}()

	if sessionID == "" {
		http.Error(w, "session id is required", http.StatusBadRequest)
		return
	}

	sess, err := a.sessRepo.GetSession(r.Context(), sessionID)
	if err != nil {
		http.Error(w, fmt.Sprintf("loading session: %s", err), http.StatusNotFound)
		return
	}

	msgs, err := loadAllMessages(r.Context(), a.msgRepo, sessionID)
	if err != nil {
		http.Error(w, fmt.Sprintf("loading messages: %s", err), http.StatusInternalServerError)
		return
	}

	view := exportView{SessionID: sessionID, CreatedAt: sess.CreatedAt, ExportedAt: time.Now(), Messages: toExportMessages(msgs)}

	var b bytes.Buffer
	if err := htmlTemplates.ExecuteTemplate(&b, "export.html", view); err != nil {
		http.Error(w, fmt.Sprintf("rendering export: %s", err), http.StatusInternalServerError)
		return
	}

	filename := fmt.Sprintf("gosimov-session-%s.html", sessionID)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filename))
	log.Printf("export ok session_id=%s messages=%d", sessionID, len(msgs))
	_, _ = w.Write(b.Bytes())
}

func (a *app) handleContextStats(w http.ResponseWriter, r *http.Request) {
	sessionID := strings.TrimSpace(r.PathValue("id"))
	if sessionID == "" {
		http.Error(w, "session id is required", http.StatusBadRequest)
		return
	}

	conv, okConv := a.conversationLogger.getStats(sessionID)
	summary, okSummary := a.summaryLogger.getStats(sessionID)

	writeJSON(w, http.StatusOK, map[string]any{"conversation": conv, "summary": summary, "has_data": okConv || okSummary})
}

func (a *app) handleSessionStatus(w http.ResponseWriter, r *http.Request) {
	sessionID := strings.TrimSpace(r.PathValue("id"))
	if sessionID == "" {
		http.Error(w, "session id is required", http.StatusBadRequest)
		return
	}

	cSession := a.getSession(sessionID)
	if cSession == nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	state := cSession.session.State()

	writeJSON(w, http.StatusOK, map[string]any{
		"session_id":    state.Session.ID,
		"created_at":    state.Session.CreatedAt,
		"running":       state.Running,
		"operation":     state.Operation,
		"turn":          state.Turn,
		"message_count": state.MessageCount,
		"usage":         state.Usage,
	})
}

func (a *app) handleSessionEvents(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	started := time.Now()
	log.Printf("events open session_id=%s remote=%s", id, r.RemoteAddr)
	defer func() {
		log.Printf("events close session_id=%s duration=%s", id, time.Since(started))
	}()

	if a.getSession(id) == nil {
		log.Printf("events missing-session session_id=%s", id)
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	events, err := a.msgRepo.Subscribe(r.Context(), subscriber.SubscribeOpts{SessionID: id, Replay: true})
	if err != nil {
		log.Printf("events subscribe error session_id=%s err=%v", id, err)
		http.Error(w, fmt.Sprintf("subscribing events: %s", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	ping := time.NewTicker(20 * time.Second)
	defer ping.Stop()
	sentMessages := 0

	for {
		select {
		case event, ok := <-events:
			if !ok {
				return
			}

			for _, msg := range event.Messages {
				payload := struct {
					SessionID string        `json:"session_id"`
					Replay    bool          `json:"replay"`
					Message   model.Message `json:"message"`
				}{SessionID: event.SessionID, Replay: event.Replay, Message: msg}

				b, err := json.Marshal(payload)
				if err != nil {
					continue
				}

				_, _ = fmt.Fprintf(w, "event: message\n")
				_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
				sentMessages++
			}
			flusher.Flush()

		case <-ping.C:
			_, _ = fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()

		case <-r.Context().Done():
			log.Printf("events context done session_id=%s sent_messages=%d", id, sentMessages)
			return
		}
	}
}

func parsePromptText(r *http.Request) (string, error) {
	contentType := r.Header.Get("Content-Type")
	if strings.HasPrefix(contentType, "application/json") {
		var req struct {
			Text string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return "", fmt.Errorf("invalid json body")
		}
		return req.Text, nil
	}

	if err := r.ParseForm(); err != nil {
		return "", fmt.Errorf("invalid form body")
	}

	return r.FormValue("text"), nil
}

func parsePromptRequest(r *http.Request) (*promptRequest, error) {
	contentType := r.Header.Get("Content-Type")
	if strings.HasPrefix(contentType, "application/json") {
		var req promptRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return nil, fmt.Errorf("invalid json body")
		}
		return &req, nil
	}

	if err := r.ParseForm(); err != nil {
		return nil, fmt.Errorf("invalid form body")
	}

	return &promptRequest{SessionID: r.FormValue("session_id"), Text: r.FormValue("text")}, nil
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func shouldRetryForExecutionEvidence(result *agent.TurnResult) bool {
	if result == nil {
		return false
	}

	hasToolResult := false
	for _, msg := range result.Messages {
		if msg.Kind == model.MessageKindToolResult {
			hasToolResult = true
			break
		}
	}

	if hasToolResult {
		return false
	}

	if result.Message.Kind != model.MessageKindLLM {
		return false
	}

	textParts := make([]string, 0, len(result.Message.Content))
	for _, p := range result.Message.Content {
		if p.Type == model.ContentPartTypeText {
			textParts = append(textParts, p.Text)
		}
	}

	text := strings.ToLower(strings.Join(textParts, "\n"))
	if strings.TrimSpace(text) == "" {
		return false
	}

	claimHints := []string{
		"i created",
		"i changed",
		"i updated",
		"i edited",
		"i implemented",
		"i ran",
		"i executed",
		"done",
		"completed",
	}

	for _, hint := range claimHints {
		if strings.Contains(text, hint) {
			return true
		}
	}

	return false
}
