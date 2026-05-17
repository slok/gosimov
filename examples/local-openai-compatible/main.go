// Example local-openai-compatible demonstrates a real agentic session using a
// custom OpenAI-compatible server such as Ollama or a local llama server.
//
// Usage:
//
//	go run ./examples/local-openai-compatible
//	go run ./examples/local-openai-compatible --base-url http://127.0.0.1:11434/v1 --model llama3.1
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/llm/customopenaicompatible"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/jsonl"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/ls"
	"github.com/slok/gosimov/pkg/tool/read"
	"github.com/slok/gosimov/pkg/tool/shell"
	"github.com/slok/gosimov/pkg/tool/write"
)

type config struct {
	baseURL         string
	apiKey          string
	modelID         string
	contextWindow   int
	maxOutputTokens int
	prompt          string
	systemPrompt    string
	maxIterations   int
	workDir         string
	storeDir        string
}

func loadConfig() (config, error) {
	var cfg config
	flag.StringVar(&cfg.baseURL, "base-url", defaultString(os.Getenv("LOCAL_OPENAI_COMPATIBLE_BASE_URL"), "http://192.168.21.241:3002/v1"), "OpenAI-compatible base URL")
	flag.StringVar(&cfg.apiKey, "api-key", os.Getenv("LOCAL_OPENAI_COMPATIBLE_API_KEY"), "Optional API key")
	flag.StringVar(&cfg.modelID, "model", defaultString(os.Getenv("LOCAL_OPENAI_COMPATIBLE_MODEL"), "Qwen3.6"), "OpenAI-compatible model ID")
	flag.IntVar(&cfg.contextWindow, "context-window", defaultInt(os.Getenv("LOCAL_OPENAI_COMPATIBLE_CONTEXT_WINDOW"), 131072), "Model context window")
	flag.IntVar(&cfg.maxOutputTokens, "max-output-tokens", defaultInt(os.Getenv("LOCAL_OPENAI_COMPATIBLE_MAX_OUTPUT_TOKENS"), 8192), "Model max output tokens")
	flag.StringVar(&cfg.prompt, "prompt", "Use the ls tool on the current directory, then tell me what you saw in one sentence.", "Prompt to execute")
	flag.StringVar(&cfg.systemPrompt, "system-prompt", "You are a helpful coding assistant. Use the available tools to complete tasks. Be concise.", "System prompt")
	flag.IntVar(&cfg.maxIterations, "max-iterations", 10, "Maximum LLM iterations per turn")
	flag.StringVar(&cfg.workDir, "work-dir", "", "Working directory for tools (defaults to temporary directory)")
	flag.StringVar(&cfg.storeDir, "store-dir", os.TempDir(), "Directory for JSONL storage")
	flag.Parse()

	cfg.baseURL = strings.TrimSpace(cfg.baseURL)
	cfg.apiKey = strings.TrimSpace(cfg.apiKey)
	cfg.modelID = strings.TrimSpace(cfg.modelID)
	cfg.prompt = strings.TrimSpace(cfg.prompt)
	cfg.systemPrompt = strings.TrimSpace(cfg.systemPrompt)
	cfg.workDir = strings.TrimSpace(cfg.workDir)
	cfg.storeDir = strings.TrimSpace(cfg.storeDir)

	if cfg.baseURL == "" {
		return config{}, fmt.Errorf("--base-url is required")
	}
	if cfg.modelID == "" {
		return config{}, fmt.Errorf("--model is required")
	}
	if cfg.contextWindow <= 0 {
		return config{}, fmt.Errorf("--context-window must be > 0")
	}
	if cfg.maxOutputTokens <= 0 {
		return config{}, fmt.Errorf("--max-output-tokens must be > 0")
	}
	if cfg.prompt == "" {
		return config{}, fmt.Errorf("--prompt is required")
	}
	if cfg.systemPrompt == "" {
		return config{}, fmt.Errorf("--system-prompt is required")
	}
	if cfg.maxIterations <= 0 {
		return config{}, fmt.Errorf("--max-iterations must be > 0")
	}
	if cfg.storeDir == "" {
		return config{}, fmt.Errorf("--store-dir is required")
	}

	return cfg, nil
}

func run(ctx context.Context) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	workDir := cfg.workDir
	cleanup := func() {}
	if workDir == "" {
		workDir, err = os.MkdirTemp("", "gosimov-local-openai-compatible-*")
		if err != nil {
			return fmt.Errorf("creating temp dir: %w", err)
		}
		cleanup = func() { _ = os.RemoveAll(workDir) }
	} else {
		if err := os.MkdirAll(workDir, 0o755); err != nil {
			return fmt.Errorf("creating work dir: %w", err)
		}
	}
	defer cleanup()

	if err := os.MkdirAll(cfg.storeDir, 0o755); err != nil {
		return fmt.Errorf("creating store dir: %w", err)
	}

	fmt.Printf("Base URL:  %s\n", cfg.baseURL)
	fmt.Printf("Model:     %s\n", cfg.modelID)
	fmt.Printf("Work dir:  %s\n\n", workDir)

	tools, err := createTools(workDir)
	if err != nil {
		return err
	}

	provider, err := customopenaicompatible.New(customopenaicompatible.Config{
		BaseURL:         cfg.baseURL,
		Model:           cfg.modelID,
		ContextWindow:   cfg.contextWindow,
		MaxOutputTokens: cfg.maxOutputTokens,
		APIKey:          cfg.apiKey,
	})
	if err != nil {
		return fmt.Errorf("creating provider: %w", err)
	}

	repo, err := jsonl.New(jsonl.Config{Dir: filepath.Clean(cfg.storeDir)})
	if err != nil {
		return fmt.Errorf("creating repository: %w", err)
	}

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      cfg.systemPrompt,
		Tools:             tools,
		SessionRepository: repo,
		MessageRepository: repo,
		TurnMaxIterations: cfg.maxIterations,
	})
	if err != nil {
		return fmt.Errorf("creating session: %w", err)
	}

	fmt.Printf("Session:   %s\n\n", session.Session().ID)
	fmt.Printf("User: %s\n\n", cfg.prompt)

	result, err := session.Prompt(ctx, []model.ContentPart{model.NewContentText(cfg.prompt)}, agent.PromptOptions{})
	if err != nil {
		return fmt.Errorf("prompt failed: %w", err)
	}

	fmt.Println("--- Turn messages ---")
	for _, msg := range result.NewMessages {
		printMessage(msg)
	}

	fmt.Println("\n--- Summary ---")
	fmt.Printf("Messages in turn: %d\n", len(result.NewMessages))
	fmt.Printf("Total messages:   %d\n", len(session.Messages()))

	usage := session.Usage()
	fmt.Printf("Tokens:           %d total, %d input, %d output\n", usage.TotalTokens, usage.InputTokens, usage.OutputTokens)

	if finalMsg := finalLLMMessage(result.NewMessages); finalMsg != nil && finalMsg.Metadata != nil {
		fmt.Printf("Model used:       %s\n", finalMsg.Metadata.Model)
		fmt.Printf("Provider:         %s\n", finalMsg.Metadata.Provider)
		fmt.Printf("Stop reason:      %s\n", finalMsg.Metadata.StopReason)
	}

	return nil
}

func createTools(workDir string) ([]tool.Tool, error) {
	lsTool, err := ls.New(ls.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating ls tool: %w", err)
	}

	readTool, err := read.New(read.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating read tool: %w", err)
	}

	writeTool, err := write.New(write.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating write tool: %w", err)
	}

	shellTool, err := shell.New(shell.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating shell tool: %w", err)
	}

	return []tool.Tool{lsTool, readTool, writeTool, shellTool}, nil
}

func printMessage(msg model.Message) {
	switch msg.Kind {
	case model.MessageKindLLM:
		if len(msg.ToolCallRequests) > 0 {
			for _, tc := range msg.ToolCallRequests {
				fmt.Printf("  LLM  -> tool call: %s(%s)\n", tc.ToolID, truncate(string(tc.Arguments), 80))
			}
		}
		if len(msg.Content) > 0 {
			text := msg.Content[0].Text
			fmt.Printf("  LLM  -> %s\n", truncate(text, 200))
		}

	case model.MessageKindToolResult:
		text := "(no content)"
		if len(msg.Content) > 0 {
			text = msg.Content[0].Text
		}
		errTag := ""
		if msg.IsError {
			errTag = " [ERROR]"
		}
		fmt.Printf("  Tool -> %s%s\n", truncate(text, 120), errTag)
	}
}

func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}

func defaultString(v, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}

	return v
}

func defaultInt(raw string, fallback int) int {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback
	}

	v, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}

	return v
}

func finalLLMMessage(messages []model.Message) *model.Message {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Kind == model.MessageKindLLM {
			return &messages[i]
		}
	}

	return nil
}

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}
