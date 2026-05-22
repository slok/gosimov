// Example mcp demonstrates a real gosimov session using MCP tools discovered
// from a remote HTTP MCP server and a real OpenCode Go provider.
//
// Usage:
//
//	go run ./examples/mcp --api-key <key> --mcp-url https://example.com/mcp
//	go run ./examples/mcp --api-key <key> --mcp-url https://example.com/mcp --mcp-header "Authorization: Bearer token"
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/slok/gosimov/examples/mcp/mcptool"
	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/opencodego"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/jsonl"
	"github.com/slok/gosimov/pkg/tool"
)

type config struct {
	apiKey        string
	modelID       string
	baseURL       string
	prompt        string
	systemPrompt  string
	maxIterations int
	storeDir      string
	mcpURL        string
	mcpNamespace  string
	mcpHeaders    headerFlags
}

func loadConfig() (config, error) {
	var cfg config

	flag.StringVar(&cfg.apiKey, "api-key", os.Getenv("OPENCODE_GO_API_KEY"), "OpenCode Go API key (or set OPENCODE_GO_API_KEY)")
	flag.StringVar(&cfg.modelID, "model", defaultString(os.Getenv("OPENCODE_GO_MODEL"), opencodego.ModelDeepseekV4Flash), "OpenCode Go model ID")
	flag.StringVar(&cfg.baseURL, "base-url", strings.TrimSpace(os.Getenv("OPENCODE_GO_BASE_URL")), "Optional OpenCode Go base URL override")
	flag.StringVar(&cfg.prompt, "prompt", "Use the available MCP tools to inspect the remote system and explain what you found.", "Prompt to execute")
	flag.StringVar(&cfg.systemPrompt, "system-prompt", "You are a helpful coding assistant. Use the available MCP tools to complete tasks. Be concise.", "System prompt")
	flag.IntVar(&cfg.maxIterations, "max-iterations", 10, "Maximum LLM iterations per turn")
	flag.StringVar(&cfg.storeDir, "store-dir", os.TempDir(), "Directory for JSONL storage")
	flag.StringVar(&cfg.mcpURL, "mcp-url", strings.TrimSpace(os.Getenv("MCP_URL")), "Remote MCP streamable HTTP endpoint URL")
	flag.StringVar(&cfg.mcpNamespace, "mcp-namespace", defaultString(os.Getenv("MCP_NAMESPACE"), "mcp"), "Namespace prefix for discovered MCP tools (tool IDs are normalized to letters, numbers, underscores, and dashes)")
	flag.Var(&cfg.mcpHeaders, "mcp-header", "Optional MCP HTTP header in 'Name: Value' form (repeatable)")
	flag.Parse()

	cfg.apiKey = strings.TrimSpace(cfg.apiKey)
	cfg.modelID = strings.TrimSpace(cfg.modelID)
	cfg.baseURL = strings.TrimSpace(cfg.baseURL)
	cfg.prompt = strings.TrimSpace(cfg.prompt)
	cfg.systemPrompt = strings.TrimSpace(cfg.systemPrompt)
	cfg.storeDir = strings.TrimSpace(cfg.storeDir)
	cfg.mcpURL = strings.TrimSpace(cfg.mcpURL)
	cfg.mcpNamespace = strings.TrimSpace(cfg.mcpNamespace)

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
	if cfg.mcpURL == "" {
		return config{}, fmt.Errorf("--mcp-url is required (or set MCP_URL)")
	}
	if cfg.mcpNamespace == "" {
		return config{}, fmt.Errorf("--mcp-namespace is required")
	}

	return cfg, nil
}

func run(ctx context.Context) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(cfg.storeDir, 0o755); err != nil {
		return fmt.Errorf("creating store dir: %w", err)
	}

	provider, err := createProvider(cfg)
	if err != nil {
		return err
	}

	mcpSession, err := connectMCP(ctx, cfg)
	if err != nil {
		return err
	}
	defer mcpSession.Close()

	toolSet, err := mcptool.NewToolSet(ctx, mcptool.ToolSetConfig{Session: mcpSession, Namespace: cfg.mcpNamespace})
	if err != nil {
		return fmt.Errorf("discovering mcp tools: %w", err)
	}

	tools := toolSet.Tools()
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
	if cfg.baseURL != "" {
		fmt.Printf("Base URL:   %s\n", cfg.baseURL)
	}
	fmt.Printf("MCP URL:    %s\n", cfg.mcpURL)
	fmt.Printf("MCP tools:  %s\n", strings.Join(toolIDs(tools), ", "))
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

func createProvider(cfg config) (llm.Provider, error) {
	provider, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource(cfg.apiKey),
		Model:       cfg.modelID,
		BaseURL:     cfg.baseURL,
	})
	if err != nil {
		return nil, fmt.Errorf("creating provider: %w", err)
	}

	return provider, nil
}

func connectMCP(ctx context.Context, cfg config) (*mcp.ClientSession, error) {
	transport := &mcp.StreamableClientTransport{
		Endpoint:             cfg.mcpURL,
		HTTPClient:           &http.Client{Transport: newStaticHeaderTransport(http.DefaultTransport, cfg.mcpHeaders.Header())},
		DisableStandaloneSSE: true,
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "gosimov-mcp-example", Version: "v1.0.0"}, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("connecting to mcp server: %w", err)
	}

	return session, nil
}

type headerFlags []string

func (h *headerFlags) String() string {
	return strings.Join(*h, ",")
}

func (h *headerFlags) Set(v string) error {
	v = strings.TrimSpace(v)
	if v == "" {
		return fmt.Errorf("header value is required")
	}

	parts := strings.SplitN(v, ":", 2)
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" {
		return fmt.Errorf("header must be in 'Name: Value' form")
	}

	*h = append(*h, v)
	return nil
}

func (h headerFlags) Header() http.Header {
	result := make(http.Header, len(h))
	for _, raw := range h {
		parts := strings.SplitN(raw, ":", 2)
		name := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		result.Add(name, value)
	}

	return result
}

type staticHeaderTransport struct {
	base    http.RoundTripper
	headers http.Header
}

func newStaticHeaderTransport(base http.RoundTripper, headers http.Header) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}

	cloned := make(http.Header, len(headers))
	for name, values := range headers {
		cloned[name] = append([]string(nil), values...)
	}

	return &staticHeaderTransport{base: base, headers: cloned}
}

func (t *staticHeaderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	for name, values := range t.headers {
		if len(values) == 0 {
			continue
		}
		for _, value := range values {
			clone.Header.Add(name, value)
		}
	}

	return t.base.RoundTrip(clone)
}

func toolIDs(tools []tool.Tool) []string {
	result := make([]string, len(tools))
	for i, tool := range tools {
		result[i] = tool.ID()
	}

	return result
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
			fmt.Printf("  LLM  -> %s\n", truncate(msg.Content[0].Text, 200))
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
