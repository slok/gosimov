// Example chat runs a small web server with a browser chat UI backed by gosimov.
//
// It demonstrates an end-to-end agentic flow with real tools:
//  1. User sends prompts from the browser.
//  2. Agent calls the LLM and executes tools autonomously.
//  3. Browser receives persisted message events over SSE using the subscriber wrapper.
//
// Usage:
//
//	go run ./examples/chat --provider zen --api-key <key>
//	go run ./examples/chat --provider opencode-go --api-key <key>
//	go run ./examples/chat --provider openai --api-key <key>
//	go run ./examples/chat --provider codex --auth-file /tmp/gosimov-chat-auth.json
//	go run ./examples/chat --provider anthropic --api-key <key>
//	go run ./examples/chat --provider claude --auth-file /tmp/gosimov-chat-auth.json
//
// Optional flags:
//   - --addr (default :8080)
//   - --provider (default zen)
//   - --api-key (required unless --auth-file is set)
//   - --auth-file (only for codex/claude providers)
//   - --model (default depends on provider)
//   - --system-prompt (optional)
//   - --max-iterations (default 100)
//   - --max-history-messages (default 60, load-session in-memory limit)
//   - --compaction-keep-recent-tokens (default 1200)
//   - --compaction-summary-model (default: same as --model)
//   - --store-dir (default /tmp/gosimov-chat-store)
//   - --work-dir (default /tmp/gosimov-chat-work)
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"github.com/slok/gosimov/pkg/agent/context/simple"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/anthropic"
	"github.com/slok/gosimov/pkg/llm/openai"
	"github.com/slok/gosimov/pkg/llm/opencodego"
	"github.com/slok/gosimov/pkg/llm/zen"
	"github.com/slok/gosimov/pkg/store/jsonl"
	"github.com/slok/gosimov/pkg/store/subscriber"
	"github.com/slok/gosimov/pkg/tool"
)

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	apiKey, tokenSrc, authStatus, err := buildAuth(cfg)
	if err != nil {
		return err
	}

	baseRepo, err := jsonl.New(jsonl.Config{Dir: cfg.storeDir})
	if err != nil {
		return fmt.Errorf("creating jsonl repository: %w", err)
	}

	msgRepo, err := subscriber.New(subscriber.Config{Repository: baseRepo})
	if err != nil {
		return fmt.Errorf("creating subscriber repository: %w", err)
	}

	schemaTools, err := createToolsForDir(cfg.workDir)
	if err != nil {
		return fmt.Errorf("creating provider tool definitions: %w", err)
	}

	providerOpenAI, err := buildProvider(cfg, apiKey, tokenSrc, cfg.modelID, schemaTools)
	if err != nil {
		return fmt.Errorf("creating provider: %w", err)
	}

	conversationLogger := &loggingProvider{name: "conversation", wrapped: providerOpenAI}
	provider := llm.Provider(conversationLogger)

	summaryProviderOpenAI, err := buildProvider(cfg, apiKey, tokenSrc, cfg.summaryModel, nil)
	if err != nil {
		return fmt.Errorf("creating summary provider: %w", err)
	}

	summaryLogger := &loggingProvider{name: "summary", wrapped: summaryProviderOpenAI}
	summaryProvider := llm.Provider(summaryLogger)

	compactor, err := simple.New(simple.Config{
		Provider:         summaryProvider,
		KeepRecentTokens: cfg.keepRecentTok,
	})
	if err != nil {
		return fmt.Errorf("creating compactor: %w", err)
	}

	a := &app{
		cfg:                cfg,
		sessRepo:           baseRepo,
		msgRepo:            msgRepo,
		provider:           provider,
		compactor:          compactor,
		conversationLogger: conversationLogger,
		summaryLogger:      summaryLogger,
		sessions:           map[string]*chatSession{},
	}

	mux := a.routes()

	fmt.Printf("Chat UI:   http://localhost%s\n", cfg.addr)
	fmt.Printf("Provider:  %s\n", cfg.provider)
	fmt.Printf("Model:     %s\n", cfg.modelID)
	fmt.Printf("Auth:      %s\n", authStatus)
	if cfg.authFile != "" {
		fmt.Printf("Auth file: %s\n", cfg.authFile)
	}
	fmt.Printf("Store dir: %s\n", cfg.storeDir)
	fmt.Printf("Work dir:  %s\n", cfg.workDir)
	fmt.Printf("Compactor: simple (keep recent tokens=%d, summary model=%s)\n", cfg.keepRecentTok, cfg.summaryModel)

	return http.ListenAndServe(cfg.addr, mux)
}

func buildProvider(cfg config, apiKey string, tokenSrc openai.TokenSource, modelID string, tools []tool.Tool) (llm.Provider, error) {
	resolveTokenSource := func(defaultFromAPIKey func(string) openai.TokenSource) (openai.TokenSource, error) {
		if tokenSrc != nil {
			return tokenSrc, nil
		}
		if apiKey == "" {
			return nil, fmt.Errorf("provider %q requires --api-key or --auth-file", cfg.provider)
		}

		return defaultFromAPIKey(apiKey), nil
	}

	switch cfg.provider {
	case providerZen:
		ts, err := resolveTokenSource(zen.NewAPIKeyTokenSource)
		if err != nil {
			return nil, err
		}
		return zen.New(zen.Config{TokenSource: ts, Model: modelID, Tools: tools})

	case providerOpenCodeGo:
		ts, err := resolveTokenSource(opencodego.NewAPIKeyTokenSource)
		if err != nil {
			return nil, err
		}
		return opencodego.New(opencodego.Config{TokenSource: ts, Model: modelID, Tools: tools})

	case providerOpenAI:
		ts, err := resolveTokenSource(openai.NewAPIKeyTokenSource)
		if err != nil {
			return nil, err
		}
		return openai.NewOpenAI(openai.OpenAIConfig{TokenSource: ts, Model: modelID, Tools: tools})

	case providerCodex:
		ts, err := resolveTokenSource(openai.NewAPIKeyTokenSource)
		if err != nil {
			return nil, err
		}
		return openai.NewChatGPT(openai.ChatGPTConfig{TokenSource: ts, Model: modelID, Tools: tools})

	case providerAnthropic:
		ts, err := resolveTokenSource(anthropic.NewAPIKeyTokenSource)
		if err != nil {
			return nil, err
		}
		return anthropic.NewAnthropic(anthropic.Config{TokenSource: ts, Model: modelID, Tools: tools})

	case providerClaude:
		ts, err := resolveTokenSource(anthropic.NewAPIKeyTokenSource)
		if err != nil {
			return nil, err
		}
		return anthropic.NewClaude(anthropic.ClaudeConfig{TokenSource: ts, Model: modelID, Tools: tools})

	default:
		return nil, fmt.Errorf("unsupported provider %q", cfg.provider)
	}
}
