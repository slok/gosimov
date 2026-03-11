package main

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/slok/gosimov/pkg/agent"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/subscriber"
)

type app struct {
	cfg       config
	sessRepo  store.SessionRepository
	msgRepo   *subscriber.Repository
	provider  llm.Provider
	compactor agentcontext.Compactor

	conversationLogger *loggingProvider
	summaryLogger      *loggingProvider

	mu       sync.RWMutex
	sessions map[string]*chatSession
}

type chatSession struct {
	session *agent.Session
	workDir string

	opMu     sync.Mutex
	opCancel context.CancelFunc
}

func (s *chatSession) startOperation(ctx context.Context) (context.Context, func()) {
	ctx, cancel := context.WithCancel(ctx)

	s.opMu.Lock()
	s.opCancel = cancel
	s.opMu.Unlock()

	finish := func() {
		s.opMu.Lock()
		if s.opCancel != nil {
			s.opCancel = nil
		}
		s.opMu.Unlock()
	}

	return ctx, finish
}

func (s *chatSession) stopOperation() (string, bool) {
	s.opMu.Lock()
	defer s.opMu.Unlock()

	if s.opCancel == nil {
		return "", false
	}

	op := string(s.session.State().Operation)
	s.opCancel()
	s.opCancel = nil

	return op, true
}

type loggingProvider struct {
	name    string
	wrapped llm.Provider

	mu    sync.RWMutex
	stats map[string]requestStats
}

type requestStats struct {
	Time            time.Time `json:"time"`
	MessageCount    int       `json:"message_count"`
	UserCount       int       `json:"user_count"`
	LLMCount        int       `json:"llm_count"`
	ToolCount       int       `json:"tool_count"`
	CompactionCount int       `json:"compaction_count"`
}

type contextKeySessionID struct{}

func (p *loggingProvider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if p.stats == nil {
		p.stats = map[string]requestStats{}
	}

	sessionID, _ := ctx.Value(contextKeySessionID{}).(string)
	if sessionID == "" {
		sessionID = "_unknown"
	}

	countByKind := map[model.MessageKind]int{}
	for _, msg := range req.Messages {
		countByKind[msg.Kind]++
	}

	stats := requestStats{
		Time:            time.Now(),
		MessageCount:    len(req.Messages),
		UserCount:       countByKind[model.MessageKindUser],
		LLMCount:        countByKind[model.MessageKindLLM],
		ToolCount:       countByKind[model.MessageKindToolResult],
		CompactionCount: countByKind[model.MessageKindCompaction],
	}
	p.mu.Lock()
	p.stats[sessionID] = stats
	p.mu.Unlock()

	log.Printf(
		"llm[%s] request messages=%d user=%d llm=%d tool=%d compaction=%d",
		p.name,
		len(req.Messages),
		countByKind[model.MessageKindUser],
		countByKind[model.MessageKindLLM],
		countByKind[model.MessageKindToolResult],
		countByKind[model.MessageKindCompaction],
	)

	resp, err := p.wrapped.Call(ctx, req)
	if err != nil {
		log.Printf("llm[%s] error: %v", p.name, err)
		return nil, err
	}

	usage := model.Usage{}
	stopReason := model.StopReasonNone
	if resp != nil && resp.Message.Metadata != nil {
		stopReason = resp.Message.Metadata.StopReason
		if resp.Message.Metadata.Usage != nil {
			usage = *resp.Message.Metadata.Usage
		}
	}

	log.Printf(
		"llm[%s] response kind=%s stop=%s usage=%d_total/%d_in/%d_out/%d_cached",
		p.name,
		resp.Message.Kind,
		stopReason,
		usage.TotalTokens,
		usage.InputTokens,
		usage.OutputTokens,
		usage.CacheReadTokens,
	)

	return resp, nil
}

func (p *loggingProvider) ModelInfo() model.LLMModelInfo {
	return p.wrapped.ModelInfo()
}

func (p *loggingProvider) getStats(sessionID string) (requestStats, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	v, ok := p.stats[sessionID]
	return v, ok
}

type promptRequest struct {
	SessionID string `json:"session_id"`
	Text      string `json:"text"`
}

type exportView struct {
	SessionID  string
	CreatedAt  time.Time
	ExportedAt time.Time
	Messages   []exportMessage
}

type exportMessage struct {
	Kind       string
	CreatedAt  time.Time
	IsError    bool
	ToolCallID string
	Text       string
	ToolCalls  []string
}
