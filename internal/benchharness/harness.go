package benchharness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/slok/gosimov/pkg/agent"
	agentcontext "github.com/slok/gosimov/pkg/agent/context/simple"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/fake"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/jsonl"
	"github.com/slok/gosimov/pkg/store/memory"
	"github.com/slok/gosimov/pkg/tool"
)

type Mode string

const (
	ModeSimple     Mode = "simple"
	ModeTools      Mode = "tools"
	ModeCompaction Mode = "compaction"
	ModeMixed      Mode = "mixed"
)

type StoreKind string

const (
	StoreMemory StoreKind = "memory"
	StoreJSONL  StoreKind = "jsonl"
)

type Config struct {
	Mode              Mode
	Sessions          int
	Turns             int
	SessionResetEvery int
	PromptBytes       int
	ResponseBytes     int
	ToolCalls         int
	ToolResultBytes   int
	CompactEvery      int
	KeepRecentTokens  int
	Store             StoreKind
	StoreDir          string
}

type Result struct {
	Sessions int
	Turns    int
}

func (c *Config) defaults() error {
	if c.Mode == "" {
		c.Mode = ModeMixed
	}

	switch c.Mode {
	case ModeSimple, ModeTools, ModeCompaction, ModeMixed:
	default:
		return fmt.Errorf("invalid mode %q", c.Mode)
	}

	if c.Sessions <= 0 {
		c.Sessions = 1
	}

	if c.Turns <= 0 {
		c.Turns = 2000
	}

	if c.PromptBytes <= 0 {
		c.PromptBytes = 256
	}

	if c.ResponseBytes <= 0 {
		c.ResponseBytes = 256
	}

	if c.ToolResultBytes <= 0 {
		c.ToolResultBytes = 512
	}

	if c.KeepRecentTokens <= 0 {
		c.KeepRecentTokens = 1200
	}

	if c.Store == "" {
		c.Store = StoreMemory
	}

	switch c.Store {
	case StoreMemory, StoreJSONL:
	default:
		return fmt.Errorf("invalid store %q", c.Store)
	}

	if c.Mode == ModeCompaction && c.CompactEvery <= 0 {
		c.CompactEvery = 50
	}

	if c.ToolCalls < 0 {
		return fmt.Errorf("tool calls must be >= 0")
	}

	if c.SessionResetEvery < 0 {
		return fmt.Errorf("session reset every must be >= 0")
	}

	if c.CompactEvery < 0 {
		return fmt.Errorf("compact every must be >= 0")
	}

	return nil
}

func Run(ctx context.Context, cfg Config) (*Result, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	sessionRepo, messageRepo, cleanup, err := repositories(cfg)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	promptBase := strings.Repeat("p", cfg.PromptBytes)

	var wg sync.WaitGroup
	errC := make(chan error, cfg.Sessions)

	for s := 0; s < cfg.Sessions; s++ {
		wg.Add(1)
		go func(sessionIdx int) {
			defer wg.Done()

			session, err := newSession(ctx, cfg, sessionRepo, messageRepo)
			if err != nil {
				errC <- err
				return
			}

			for turn := 0; turn < cfg.Turns; turn++ {
				if err := ctx.Err(); err != nil {
					errC <- err
					return
				}

				prompt := makePrompt(promptBase, sessionIdx, turn)

				_, err := session.Prompt(ctx, []model.ContentPart{model.NewContentText(prompt)}, agent.PromptOptions{})
				if err != nil {
					errC <- fmt.Errorf("session %d turn %d prompt: %w", sessionIdx, turn, err)
					return
				}

				if cfg.CompactEvery > 0 && (turn+1)%cfg.CompactEvery == 0 {
					_, err := session.Compact(ctx)
					if err != nil {
						errC <- fmt.Errorf("session %d turn %d compact: %w", sessionIdx, turn, err)
						return
					}
				}

				if cfg.SessionResetEvery > 0 && (turn+1)%cfg.SessionResetEvery == 0 && turn != cfg.Turns-1 {
					session, err = newSession(ctx, cfg, sessionRepo, messageRepo)
					if err != nil {
						errC <- fmt.Errorf("session %d turn %d reset: %w", sessionIdx, turn, err)
						return
					}
				}
			}
		}(s)
	}

	wg.Wait()
	close(errC)

	for err := range errC {
		if err != nil {
			return nil, err
		}
	}

	return &Result{Sessions: cfg.Sessions, Turns: cfg.Sessions * cfg.Turns}, nil
}

func repositories(cfg Config) (store.SessionRepository, store.MessageRepository, func(), error) {
	switch cfg.Store {
	case StoreMemory:
		repo := memory.NewRepository()
		return repo, repo, func() {}, nil

	case StoreJSONL:
		dir := strings.TrimSpace(cfg.StoreDir)
		cleanup := func() {}
		if dir == "" {
			tmpDir, err := os.MkdirTemp("", "gosimov-benchharness-*")
			if err != nil {
				return nil, nil, nil, fmt.Errorf("creating temp store dir: %w", err)
			}
			dir = tmpDir
			cleanup = func() { _ = os.RemoveAll(tmpDir) }
		}

		repo, err := jsonl.New(jsonl.Config{Dir: dir})
		if err != nil {
			cleanup()
			return nil, nil, nil, fmt.Errorf("creating jsonl repository: %w", err)
		}

		return repo, repo, cleanup, nil

	default:
		return nil, nil, nil, fmt.Errorf("unsupported store %q", cfg.Store)
	}
}

func (c Config) toolCallsForTurn(turn int) int {
	defaultToolCalls := 2
	if c.ToolCalls > 0 {
		defaultToolCalls = c.ToolCalls
	}

	switch c.Mode {
	case ModeSimple:
		return 0
	case ModeTools:
		return defaultToolCalls
	case ModeCompaction:
		return defaultToolCalls
	case ModeMixed:
		if turn%2 == 0 {
			return 0
		}
		return defaultToolCalls
	default:
		return defaultToolCalls
	}
}

func newSession(ctx context.Context, cfg Config, sessionRepo store.SessionRepository, messageRepo store.MessageRepository) (*agent.Session, error) {
	toolCallTool := payloadTool{payload: strings.Repeat("t", cfg.ToolResultBytes)}

	compactor, err := agentcontext.New(agentcontext.Config{
		Provider:         summaryProvider(),
		ReserveTokens:    256,
		KeepRecentTokens: cfg.KeepRecentTokens,
		MaxSummaryTokens: 128,
	})
	if err != nil {
		return nil, fmt.Errorf("creating compactor: %w", err)
	}

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          scriptedProvider(cfg),
		Tools:             []tool.Tool{toolCallTool},
		Compactor:         compactor,
		SessionRepository: sessionRepo,
		MessageRepository: messageRepo,
	})
	if err != nil {
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return session, nil
}

func makePrompt(base string, sessionIdx, turn int) string {
	return fmt.Sprintf("[session=%d] [turn=%d] %s", sessionIdx, turn, base)
}

func scriptedProvider(cfg Config) llm.Provider {
	responseText := strings.Repeat("r", cfg.ResponseBytes)

	return fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
		desiredToolCalls := cfg.toolCallsForTurn(activeTurnIndex(req.Messages))
		currentToolResults := toolResultsInActiveTurn(req.Messages)

		if currentToolResults < desiredToolCalls {
			return &llm.Response{Message: model.Message{
				Kind: model.MessageKindLLM,
				ToolCallRequests: []model.ToolCallRequest{{
					ID:        fmt.Sprintf("tc-%d", currentToolResults+1),
					ToolID:    payloadToolID,
					Arguments: json.RawMessage(`{}`),
				}},
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonToolUse,
					Usage:      &model.Usage{InputTokens: 64, OutputTokens: 32},
				},
			}}, nil
		}

		return &llm.Response{Message: model.Message{
			Kind:    model.MessageKindLLM,
			Content: []model.ContentPart{model.NewContentText(responseText)},
			Metadata: &model.MessageMetadata{
				StopReason: model.StopReasonComplete,
				Usage:      &model.Usage{InputTokens: 64, OutputTokens: 32},
			},
		}}, nil
	})
}

func summaryProvider() llm.Provider {
	return fake.NewProviderWithModelInfo(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
		return &llm.Response{Message: model.Message{
			Kind:    model.MessageKindLLM,
			Content: []model.ContentPart{model.NewContentText("checkpoint summary")},
			Metadata: &model.MessageMetadata{
				StopReason: model.StopReasonComplete,
				Usage:      &model.Usage{InputTokens: 32, OutputTokens: 16},
			},
		}}, nil
	}, model.LLMModelInfo{ID: "fake-summary", ContextWindow: 2048, MaxOutputTokens: 256})
}

func activeTurnIndex(messages []model.Message) int {
	turn := -1
	for _, msg := range messages {
		if msg.Kind == model.MessageKindUser {
			turn++
		}
	}

	if turn < 0 {
		return 0
	}

	return turn
}

func toolResultsInActiveTurn(messages []model.Message) int {
	lastUser := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Kind == model.MessageKindUser {
			lastUser = i
			break
		}
	}

	if lastUser == -1 {
		return 0
	}

	count := 0
	for _, msg := range messages[lastUser+1:] {
		if msg.Kind == model.MessageKindToolResult {
			count++
		}
	}

	return count
}

const payloadToolID = "fake_payload"

type payloadTool struct {
	payload string
}

func (t payloadTool) ID() string { return payloadToolID }

func (t payloadTool) Description() string { return "Returns deterministic payload text." }

func (t payloadTool) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","additionalProperties":false}`)
}

func (t payloadTool) Execute(_ context.Context, _ json.RawMessage) (*tool.Result, error) {
	return &tool.Result{Content: []model.ContentPart{model.NewContentText(t.payload)}}, nil
}
