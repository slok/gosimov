// Example skills demonstrates a local skill-loading tool with OpenCode Go.
//
// Usage:
//
//	go run ./examples/skills --api-key <key>
//	go run ./examples/skills --api-key <key> --skills-dir ./examples/skills/skills
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	skilltools "github.com/slok/gosimov/examples/skills/tools"
	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/llm/opencodego"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/jsonl"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/ls"
	"github.com/slok/gosimov/pkg/tool/read"
	"github.com/slok/gosimov/pkg/tool/shell"
	"github.com/slok/gosimov/pkg/tool/write"
)

type config struct {
	apiKey        string
	modelID       string
	prompt        string
	systemPrompt  string
	debug         bool
	maxIterations int
	workDir       string
	storeDir      string
	skillsDir     string
}

func loadConfig() (config, error) {
	var cfg config

	flag.StringVar(&cfg.apiKey, "api-key", os.Getenv("OPENCODE_GO_API_KEY"), "OpenCode Go API key (or set OPENCODE_GO_API_KEY)")
	flag.StringVar(&cfg.modelID, "model", defaultString(os.Getenv("OPENCODE_GO_MODEL"), opencodego.ModelDeepseekV4Flash), "OpenCode Go model ID")
	flag.StringVar(&cfg.prompt, "prompt", "Prepare release notes for this repository. Use available skills when relevant.", "Prompt to execute")
	flag.StringVar(&cfg.systemPrompt, "system-prompt", "You are a helpful coding assistant. Prefer loading a relevant skill before running task-specific workflows.", "System prompt")
	flag.BoolVar(&cfg.debug, "debug", false, "Enable debug logs for tool execution")
	flag.IntVar(&cfg.maxIterations, "max-iterations", 10, "Maximum LLM iterations per turn")
	flag.StringVar(&cfg.workDir, "work-dir", "", "Working directory for tools (defaults to temporary directory)")
	flag.StringVar(&cfg.storeDir, "store-dir", os.TempDir(), "Directory for JSONL storage")
	flag.StringVar(&cfg.skillsDir, "skills-dir", "examples/skills/skills", "Directory that contains skill folders with SKILL.md files")
	flag.Parse()

	cfg.apiKey = strings.TrimSpace(cfg.apiKey)
	cfg.modelID = strings.TrimSpace(cfg.modelID)
	cfg.prompt = strings.TrimSpace(cfg.prompt)
	cfg.systemPrompt = strings.TrimSpace(cfg.systemPrompt)
	cfg.workDir = strings.TrimSpace(cfg.workDir)
	cfg.storeDir = strings.TrimSpace(cfg.storeDir)
	cfg.skillsDir = strings.TrimSpace(cfg.skillsDir)

	if cfg.apiKey == "" {
		return config{}, fmt.Errorf("--api-key is required (or set OPENCODE_GO_API_KEY)")
	}

	if cfg.modelID == "" {
		return config{}, fmt.Errorf("--model is required")
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

	if cfg.skillsDir == "" {
		return config{}, fmt.Errorf("--skills-dir is required")
	}

	return cfg, nil
}

func run(ctx context.Context) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	skillsDir, err := filepath.Abs(cfg.skillsDir)
	if err != nil {
		return fmt.Errorf("resolve skills dir path: %w", err)
	}

	workDir := cfg.workDir
	cleanup := func() {}
	if workDir == "" {
		workDir, err = os.MkdirTemp("", "gosimov-skills-*")
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

	tools, err := createTools(workDir, skillsDir, cfg.debug)
	if err != nil {
		return err
	}

	provider, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource(cfg.apiKey),
		Model:       cfg.modelID,
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

	fmt.Printf("Model:      %s\n", cfg.modelID)
	fmt.Printf("Skills dir: %s\n", skillsDir)
	fmt.Printf("Work dir:   %s\n", workDir)
	fmt.Printf("Debug:      %t\n", cfg.debug)
	fmt.Printf("Session:    %s\n\n", session.Session().ID)
	fmt.Printf("User: %s\n\n", cfg.prompt)

	result, err := session.Prompt(ctx, []model.ContentPart{model.NewContentText(cfg.prompt)}, agent.PromptOptions{})
	if err != nil {
		return fmt.Errorf("prompt failed: %w", err)
	}

	fmt.Println("--- Turn messages ---")
	for _, msg := range result.NewMessages {
		printMessage(msg)
	}

	finalMessage := ""
	if finalMsg := finalLLMMessage(result.NewMessages); finalMsg != nil {
		finalMessage = firstText(*finalMsg)
	}
	if finalMessage == "" {
		finalMessage = "(no text content in final LLM message)"
	}
	fmt.Printf("\nFinal LLM message:\n%s\n", finalMessage)

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

func createTools(workDir string, skillsDir string, debug bool) ([]tool.Tool, error) {
	wrap := func(t tool.Tool) tool.Tool { return skilltools.WrapWithLogging(t, debug) }

	skillTool, err := skilltools.NewSkillTool(skilltools.SkillToolConfig{SkillsDir: skillsDir})
	if err != nil {
		return nil, fmt.Errorf("creating skill tool: %w", err)
	}

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

	return []tool.Tool{skillTool, wrap(lsTool), wrap(readTool), wrap(writeTool), wrap(shellTool)}, nil
}

func printMessage(msg model.Message) {
	switch msg.Kind {
	case model.MessageKindLLM:
		if len(msg.ToolCallRequests) > 0 {
			for _, tc := range msg.ToolCallRequests {
				fmt.Printf("  LLM  -> tool call: %s(%s)\n", tc.ToolID, truncate(string(tc.Arguments), 80))
			}
		}

		if text := firstText(msg); text != "" {
			fmt.Printf("  LLM  -> %s\n", truncate(text, 200))
		}

	case model.MessageKindToolResult:
		text := firstText(msg)
		if text == "" {
			text = "(no content)"
		}

		errTag := ""
		if msg.IsError {
			errTag = " [ERROR]"
		}

		fmt.Printf("  Tool -> %s%s\n", truncate(text, 140), errTag)
	}
}

func firstText(msg model.Message) string {
	for _, p := range msg.Content {
		if p.Type == model.ContentPartTypeText && strings.TrimSpace(p.Text) != "" {
			return p.Text
		}
	}

	return ""
}

func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	if len(s) > max {
		return s[:max] + "..."
	}

	return s
}

func defaultString(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}

	return ""
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
