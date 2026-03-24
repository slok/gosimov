package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/slok/gosimov/internal/utils/id"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/conventions"
	"github.com/slok/gosimov/pkg/llm"
	gosimovlog "github.com/slok/gosimov/pkg/log"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/tool"
)

// SessionConfig configures a [Session].
type SessionConfig struct {
	// Provider is the LLM to call (required).
	Provider llm.Provider
	// SystemPrompt is sent to the LLM as the system instruction (optional).
	SystemPrompt string
	// Tools available for the LLM to call (optional).
	Tools []tool.Tool
	// ToolTimeout limits each individual tool execution.
	// 0 means no timeout.
	ToolTimeout time.Duration
	// TurnMaxIterations limits how many LLM calls each turn can make.
	// 0 means no limit.
	TurnMaxIterations int
	// DisablePromptCache disables provider-side prompt caching.
	// By default prompt caching is enabled.
	DisablePromptCache bool
	// SessionRepository persists session identity (required).
	// The session is stored on creation via [store.SessionRepository.CreateSession].
	SessionRepository store.SessionRepository
	// MessageRepository persists messages (required).
	// Each message is persisted individually as it is produced:
	// user messages before the turn, LLM responses and tool results during the turn.
	MessageRepository store.MessageRepository
	// Messages preloads initial history (advanced customization, optional).
	//
	// Most callers should leave this nil and start from an empty conversation.
	// Set this when creating a branched session from existing history.
	//
	// When set, the session copies these messages, reconstructs usage from their
	// metadata, and persists them on creation.
	Messages []model.Message
	// Compactor manages context compaction within the agent loop (optional).
	// If set, it runs on every LLM call within a turn before the context processor.
	// It may create compaction checkpoints and filters messages based on those checkpoints.
	// Implementations must treat messages as immutable.
	Compactor agentcontext.Compactor
	// ContextProcessor transforms messages before each LLM call (optional).
	// If set, it is called on every LLM call within a turn (including iterations
	// after tool results), after the compactor. Implementations must treat
	// messages as immutable and return a new slice when transforming.
	ContextProcessor agentcontext.Processor
	// Logger records session and turn lifecycle events (optional).
	// If nil, [log.Noop] is used.
	Logger gosimovlog.Logger
}

func (c *SessionConfig) defaults() error {
	if c.Provider == nil {
		return fmt.Errorf("provider is required: %w", pkgerrors.ErrNotValid)
	}

	if c.SessionRepository == nil {
		return fmt.Errorf("session repository is required: %w", pkgerrors.ErrNotValid)
	}

	if c.MessageRepository == nil {
		return fmt.Errorf("message repository is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Compactor == nil {
		c.Compactor = agentcontext.NoopCompactor{}
	}

	if c.Logger == nil {
		c.Logger = gosimovlog.Noop
	}

	return nil
}

// LoadSessionConfig configures loading an existing persisted [Session].
type LoadSessionConfig struct {
	// SessionID identifies the existing session to load (required).
	SessionID string
	// Provider is the LLM to call (required).
	Provider llm.Provider
	// SystemPrompt is sent to the LLM as the system instruction (optional).
	SystemPrompt string
	// Tools available for the LLM to call (optional).
	Tools []tool.Tool
	// ToolTimeout limits each individual tool execution.
	// 0 means no timeout.
	ToolTimeout time.Duration
	// TurnMaxIterations limits how many LLM calls each turn can make.
	// 0 means no limit.
	TurnMaxIterations int
	// DisablePromptCache disables provider-side prompt caching.
	// By default prompt caching is enabled.
	DisablePromptCache bool
	// SessionRepository is used to load the existing session identity (required).
	SessionRepository store.SessionRepository
	// MessageRepository is required and used to preload history when Messages is empty.
	MessageRepository store.MessageRepository
	// Messages overrides repository preloading when non-empty (advanced customization, optional).
	//
	// Most callers should leave this nil and let MessageRepository preload the
	// persisted conversation.
	//
	// When non-empty, these messages are copied and used as the in-memory history
	// instead of loading from MessageRepository.
	Messages []model.Message
	// Compactor manages context compaction within the agent loop (optional).
	Compactor agentcontext.Compactor
	// ContextProcessor transforms messages before each LLM call (optional).
	ContextProcessor agentcontext.Processor
	// Logger records session and turn lifecycle events (optional).
	// If nil, [log.Noop] is used.
	Logger gosimovlog.Logger
}

// PromptOptions configures per-call overrides for [Session.Prompt] and [Session.Continue].
//
// Zero values use the session defaults:
//   - empty SystemPrompt uses [SessionConfig.SystemPrompt].
//   - TurnMaxIterations == 0 uses [SessionConfig.TurnMaxIterations].
type PromptOptions struct {
	// SystemPrompt overrides the session SystemPrompt for this call only.
	// Empty string uses [SessionConfig.SystemPrompt].
	SystemPrompt string
	// TurnMaxIterations overrides the session TurnMaxIterations for this call only.
	// 0 uses [SessionConfig.TurnMaxIterations].
	TurnMaxIterations int
}

// SessionTurnResult is the result returned by [Session.Prompt] and [Session.Continue].
type SessionTurnResult struct {
	// NewMessages contains only messages generated by the turn runner.
	//
	// For [Session.Prompt], this does NOT include the user message created from the
	// prompt input. The user message is persisted and appended to session history,
	// but the returned result includes only turn-generated messages.
	NewMessages []model.Message
}

// SessionOperation identifies which session operation is currently running.
type SessionOperation string

const (
	SessionOperationNone     SessionOperation = ""
	SessionOperationPrompt   SessionOperation = "prompt"
	SessionOperationContinue SessionOperation = "continue"
	SessionOperationCompact  SessionOperation = "compact"
)

// SessionState is a snapshot of the current session runtime state.
type SessionState struct {
	Session      model.Session
	Running      bool
	Operation    SessionOperation
	Turn         int
	MessageCount int
	Usage        model.Usage
}

func (c *LoadSessionConfig) defaults() error {
	if c.SessionID == "" {
		return fmt.Errorf("session id is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Provider == nil {
		return fmt.Errorf("provider is required: %w", pkgerrors.ErrNotValid)
	}

	if c.SessionRepository == nil {
		return fmt.Errorf("session repository is required: %w", pkgerrors.ErrNotValid)
	}

	if c.MessageRepository == nil {
		return fmt.Errorf("message repository is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Compactor == nil {
		c.Compactor = agentcontext.NoopCompactor{}
	}

	if c.Logger == nil {
		c.Logger = gosimovlog.Noop
	}

	return nil
}

// Session manages a multi-turn conversation with an LLM.
//
// It accumulates messages across turns, tracks total usage, and delegates
// each turn to [runTurn]. A session is the stateful wrapper that makes
// multi-turn conversations ergonomic.
//
// Session is safe for concurrent access, but only one [Prompt] or [Continue]
// call can be active at a time — concurrent calls return [ErrSessionBusy].
type Session struct {
	mu                 sync.Mutex
	session            model.Session
	provider           llm.Provider
	systemPrompt       string
	disablePromptCache bool
	tools              []tool.Tool
	toolIndex          map[string]tool.Tool
	toolTimeout        time.Duration
	maxIterations      int
	messages           []model.Message
	usage              model.Usage
	running            bool
	runningOperation   SessionOperation
	sessionRepo        store.SessionRepository
	messageRepo        store.MessageRepository
	compactor          agentcontext.Compactor
	contextProcessor   agentcontext.Processor
	logger             gosimovlog.Logger
}

// NewSession creates a new session.
//
// Session identity is always persisted on creation.
func NewSession(ctx context.Context, cfg SessionConfig) (*Session, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid session config: %w", err)
	}

	toolIndex, err := buildToolIndex(cfg.Tools)
	if err != nil {
		return nil, fmt.Errorf("invalid session config: %w", err)
	}

	sess := model.Session{
		ID:        id.NewULID(conventions.IDPrefixSession),
		CreatedAt: time.Now(),
	}

	if err := cfg.SessionRepository.CreateSession(ctx, sess); err != nil {
		return nil, fmt.Errorf("persisting session: %w", err)
	}

	initialMessages := cloneMessages(cfg.Messages)
	if len(initialMessages) > 0 {
		if err := cfg.MessageRepository.StoreMessages(ctx, sess.ID, initialMessages); err != nil {
			return nil, fmt.Errorf("persisting initial messages: %w", err)
		}
	}

	s := &Session{
		session:            sess,
		provider:           cfg.Provider,
		systemPrompt:       cfg.SystemPrompt,
		disablePromptCache: cfg.DisablePromptCache,
		tools:              cfg.Tools,
		toolIndex:          toolIndex,
		toolTimeout:        cfg.ToolTimeout,
		maxIterations:      cfg.TurnMaxIterations,
		messages:           initialMessages,
		usage:              usageFromMessages(initialMessages),
		sessionRepo:        cfg.SessionRepository,
		messageRepo:        cfg.MessageRepository,
		compactor:          cfg.Compactor,
		contextProcessor:   cfg.ContextProcessor,
		logger:             cfg.Logger,
	}

	return s, nil
}

// LoadSession loads an existing persisted session.
//
// The session identity is loaded from SessionRepository.
//
// If [LoadSessionConfig.Messages] is non-empty, those messages are copied and used
// as the in-memory history. Otherwise, history is loaded from MessageRepository.
func LoadSession(ctx context.Context, cfg LoadSessionConfig) (*Session, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid load session config: %w", err)
	}

	toolIndex, err := buildToolIndex(cfg.Tools)
	if err != nil {
		return nil, fmt.Errorf("invalid load session config: %w", err)
	}

	existing, err := cfg.SessionRepository.GetSession(ctx, cfg.SessionID)
	if err != nil {
		return nil, fmt.Errorf("loading existing session: %w", err)
	}

	s := &Session{
		session:            *existing,
		provider:           cfg.Provider,
		systemPrompt:       cfg.SystemPrompt,
		disablePromptCache: cfg.DisablePromptCache,
		tools:              cfg.Tools,
		toolIndex:          toolIndex,
		toolTimeout:        cfg.ToolTimeout,
		maxIterations:      cfg.TurnMaxIterations,
		sessionRepo:        cfg.SessionRepository,
		messageRepo:        cfg.MessageRepository,
		compactor:          cfg.Compactor,
		contextProcessor:   cfg.ContextProcessor,
		logger:             cfg.Logger,
	}

	if len(cfg.Messages) > 0 {
		s.messages = cloneMessages(cfg.Messages)
		s.usage = usageFromMessages(s.messages)
		return s, nil
	}

	msgs, err := listAllMessages(ctx, cfg.MessageRepository, existing.ID)
	if err != nil {
		return nil, fmt.Errorf("loading existing messages: %w", err)
	}

	s.messages = msgs
	s.usage = usageFromMessages(msgs)

	return s, nil
}

// buildToolIndex creates a map from tool ID to tool for O(1) lookup.
// Returns an error if duplicate tool IDs are found.
func buildToolIndex(tools []tool.Tool) (map[string]tool.Tool, error) {
	index := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolID := t.ID()
		if _, ok := index[toolID]; ok {
			return nil, fmt.Errorf("duplicate tool id %q: %w", toolID, pkgerrors.ErrNotValid)
		}
		index[toolID] = t
	}
	return index, nil
}

func listAllMessages(ctx context.Context, repo store.MessageRepository, sessionID string) ([]model.Message, error) {
	all := []model.Message{}
	opts := store.ListOpts{Limit: 100}

	for {
		result, err := repo.ListMessages(ctx, sessionID, opts)
		if err != nil {
			return nil, err
		}

		all = append(all, result.Items...)

		if result.NextCursor == "" {
			return all, nil
		}

		opts.Cursor = result.NextCursor
	}
}

func usageFromMessages(messages []model.Message) model.Usage {
	total := model.Usage{}

	for _, msg := range messages {
		total = addUsage(total, msg.Metadata)
	}

	return total
}

func cloneMessages(messages []model.Message) []model.Message {
	if messages == nil {
		return nil
	}

	cloned := make([]model.Message, len(messages))
	copy(cloned, messages)

	return cloned
}

// Prompt sends a user message and runs a full turn.
//
// It builds a [model.MessageKindUser] message from the given content,
// appends it to the conversation, runs a turn via [runTurn], and
// appends all generated messages to the session history.
func (s *Session) Prompt(ctx context.Context, content []model.ContentPart, opts PromptOptions) (*SessionTurnResult, error) {
	if err := s.beginRun(SessionOperationPrompt); err != nil {
		return nil, err
	}
	defer s.endRun()

	logger := s.sessionLogger(SessionOperationPrompt)
	logger.Debugf("Starting prompt turn")

	// Build the user message.
	userMsg := model.Message{
		ID:        id.NewULID(conventions.IDPrefixMessageUser),
		Kind:      model.MessageKindUser,
		Content:   content,
		CreatedAt: time.Now(),
	}

	// Persist user message eagerly (before running the turn).
	if s.messageRepo != nil {
		if err := s.messageRepo.StoreMessages(ctx, s.session.ID, []model.Message{userMsg}); err != nil {
			return nil, fmt.Errorf("persisting user message: %w", err)
		}
	}

	s.mu.Lock()
	s.messages = append(s.messages, userMsg)
	messages := s.messages
	s.mu.Unlock()

	result, err := s.runTurn(ctx, messages, opts)
	if err != nil {
		return nil, err
	}

	logger.WithValues(gosimovlog.KV{"turn_messages": len(result.NewMessages)}).Debugf("Prompt turn ended")

	return result, nil
}

// Continue resumes the conversation from the current message history.
//
// Use this for retries after errors. It calls [runTurn] with the current
// messages without adding a new user message.
func (s *Session) Continue(ctx context.Context, opts PromptOptions) (*SessionTurnResult, error) {
	if err := s.beginRun(SessionOperationContinue); err != nil {
		return nil, err
	}
	defer s.endRun()

	logger := s.sessionLogger(SessionOperationContinue)
	logger.Debugf("Continuing session turn")

	s.mu.Lock()
	if len(s.messages) == 0 {
		s.mu.Unlock()
		return nil, fmt.Errorf("cannot continue: no messages in session: %w", pkgerrors.ErrNotValid)
	}

	messages := s.messages
	s.mu.Unlock()

	result, err := s.runTurn(ctx, messages, opts)
	if err != nil {
		return nil, err
	}

	logger.WithValues(gosimovlog.KV{"turn_messages": len(result.NewMessages)}).Debugf("Session continuation ended")

	return result, nil
}

// runTurn executes a turn and updates session state with the results.
//
// Each message produced during the turn (LLM responses and tool results) is
// persisted individually via the onMessages callback as it is created.

func (s *Session) runTurn(ctx context.Context, messages []model.Message, opts PromptOptions) (*SessionTurnResult, error) {
	ctx = s.ctxWithRuntimeInfo(ctx)

	systemPrompt := s.systemPrompt
	if opts.SystemPrompt != "" {
		systemPrompt = opts.SystemPrompt
	}

	maxIterations := s.maxIterations
	if opts.TurnMaxIterations > 0 {
		maxIterations = opts.TurnMaxIterations
	}

	runnerResult, err := runTurn(ctx, turnConfig{
		session:            s.session,
		provider:           s.provider,
		systemPrompt:       systemPrompt,
		disablePromptCache: s.disablePromptCache,
		messages:           messages,
		tools:              s.tools,
		toolIndex:          s.toolIndex,
		toolTimeout:        s.toolTimeout,
		maxIterations:      maxIterations,
		onMessages:         s.persistMessages,
		compactor:          s.compactor,
		contextProcessor:   s.contextProcessor,
		logger:             s.sessionLogger(""),
	})
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	s.messages = append(s.messages, runnerResult.Messages...)
	s.usage = addUsage(s.usage, &model.MessageMetadata{Usage: &runnerResult.Usage})
	s.mu.Unlock()

	return &SessionTurnResult{NewMessages: runnerResult.Messages}, nil
}

// persistMessages is the onMessagesFn callback that persists messages to the store.
// It is only wired when a MessageRepository is configured.
func (s *Session) persistMessages(ctx context.Context, msgs []model.Message) error {
	if s.messageRepo == nil {
		return nil
	}

	return s.messageRepo.StoreMessages(ctx, s.session.ID, msgs)
}

// Messages returns a copy of the conversation history.
func (s *Session) Messages() []model.Message {
	s.mu.Lock()
	defer s.mu.Unlock()

	msgs := make([]model.Message, len(s.messages))
	copy(msgs, s.messages)

	return msgs
}

// Session returns the session identity (ID and creation time).
func (s *Session) Session() model.Session {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.session
}

// Usage returns the aggregated token usage across all turns in the session.
func (s *Session) Usage() model.Usage {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.usage
}

// State returns a thread-safe snapshot of the session runtime state.
func (s *Session) State() SessionState {
	s.mu.Lock()
	session := s.session
	usage := s.usage
	running := s.running
	operation := s.runningOperation
	messages := make([]model.Message, len(s.messages))
	copy(messages, s.messages)
	s.mu.Unlock()

	return SessionState{
		Session:      session,
		Running:      running,
		Operation:    operation,
		Turn:         len(model.TurnsFromMessages(messages)),
		MessageCount: len(messages),
		Usage:        usage,
	}
}

// Compact forces a context compaction between turns.
//
// It delegates to [runCompaction] with [CompactOptions.Force] set to true.
// If the compactor creates a compaction checkpoint message, it is appended to the
// conversation history and persisted (persistence happens inside [runCompaction]).
//
// Returns the [CompactResult] so the caller can inspect the compaction message
// and usage. If no compactor is configured (NoopCompactor), the result will have
// a nil Message and the full message list.
//
// Cannot be called while a turn is running — returns [ErrSessionBusy].
func (s *Session) Compact(ctx context.Context) (*agentcontext.CompactResult, error) {
	if err := s.beginRun(SessionOperationCompact); err != nil {
		return nil, err
	}
	defer s.endRun()

	logger := s.sessionLogger(SessionOperationCompact)
	logger.Debugf("Starting message compaction...")

	ctx = s.ctxWithRuntimeInfo(ctx)

	s.mu.Lock()
	messages := s.messages
	s.mu.Unlock()

	runtimeMessages := effectiveCompactionContext(messages)

	result, err := runCompaction(ctx, compactionConfig{
		compactor:  s.compactor,
		messages:   runtimeMessages,
		onMessages: s.persistMessages,
		opts:       agentcontext.CompactOptions{Force: true},
		logger:     logger,
	})
	if err != nil {
		return nil, err
	}

	if result.SummaryMessage != nil {
		s.mu.Lock()
		s.messages = append(s.messages, *result.SummaryMessage)
		s.usage = addUsage(s.usage, &model.MessageMetadata{Usage: &result.Usage})
		s.mu.Unlock()

		logger.Debugf("Compaction succeeded, checkpoint message appended to conversation history")
	} else {
		logger.Debugf("No compaction happened, conversation history left unchanged")
	}

	return result, nil
}

func (s *Session) sessionLogger(op SessionOperation) gosimovlog.Logger {
	logger := s.logger

	kv := gosimovlog.KV{
		"component":  "agent.session",
		"session_id": s.session.ID,
	}

	if op != SessionOperationNone {
		kv["operation"] = string(op)
	}

	return logger.WithValues(kv)
}

func (s *Session) ctxWithRuntimeInfo(parent context.Context) context.Context {
	ctx := ctxWithSessionID(parent, s.session.ID)
	modelInfo := s.provider.ModelInfo()
	return ctxWithLLMModelInfo(ctx, &modelInfo)
}

func (s *Session) beginRun(op SessionOperation) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return pkgerrors.ErrSessionBusy
	}

	s.running = true
	s.runningOperation = op

	return nil
}

func (s *Session) endRun() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.running = false
	s.runningOperation = SessionOperationNone
}
