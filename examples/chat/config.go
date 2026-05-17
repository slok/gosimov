package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/slok/gosimov/pkg/llm/openai"
)

type providerKind string

const (
	providerZen        providerKind = "zen"
	providerOpenCodeGo providerKind = "opencode-go"
	providerOpenAI     providerKind = "openai"
	providerCodex      providerKind = "codex"
)

const (
	defaultOAuthAuthFile = "/tmp/gosimov-chat-auth.json"

	openAIOAuthStoreKey = "openai"
	openAIClientID      = "app_EMoamEEZ73f0CkXaXp7hrann"
	openAIAuthURL       = "https://auth.openai.com/oauth/authorize"
	openAITokenURL      = "https://auth.openai.com/oauth/token"
	openAIRedirectURL   = "http://localhost:1455/auth/callback"
	openAIScopes        = "openid profile email offline_access"
)

var defaultModelByProvider = map[providerKind]string{
	providerZen:        "glm-5-free",
	providerOpenCodeGo: "deepseek-v4-flash",
	providerOpenAI:     "gpt-5",
	providerCodex:      "gpt-5.3-codex",
}

type config struct {
	addr          string
	debug         bool
	provider      providerKind
	apiKey        string
	authFile      string
	modelID       string
	systemPrompt  string
	maxIterations int
	maxHistory    int
	keepRecentTok int
	summaryModel  string
	storeDir      string
	workDir       string
}

func loadConfig() (config, error) {
	var provider string
	var cfg config
	flag.StringVar(&cfg.addr, "addr", ":8080", "HTTP listen address")
	flag.BoolVar(&cfg.debug, "debug", false, "Enable debug logs")
	flag.StringVar(&provider, "provider", string(providerZen), "LLM provider: zen|opencode-go|openai|codex")
	flag.StringVar(&cfg.apiKey, "api-key", "", "Provider API key/token (required unless --auth-file is set)")
	flag.StringVar(&cfg.authFile, "auth-file", "", "OAuth credentials file path (only for codex provider)")
	flag.StringVar(&cfg.modelID, "model", "", "LLM model ID (defaults depend on --provider)")
	flag.StringVar(&cfg.systemPrompt, "system-prompt", "", "System prompt for new sessions")
	flag.IntVar(&cfg.maxIterations, "max-iterations", 100, "Maximum LLM iterations per turn")
	flag.IntVar(&cfg.maxHistory, "max-history-messages", 60, "Maximum number of historical messages kept in memory when loading an existing session (0 means unlimited)")
	flag.IntVar(&cfg.keepRecentTok, "compaction-keep-recent-tokens", 1200, "Compaction keep-recent token target")
	flag.StringVar(&cfg.summaryModel, "compaction-summary-model", "", "Model used by compactor summarization (defaults to --model)")
	flag.StringVar(&cfg.storeDir, "store-dir", filepath.Join(os.TempDir(), "gosimov-chat-store"), "Directory for JSONL session/message storage")
	flag.StringVar(&cfg.workDir, "work-dir", filepath.Join(os.TempDir(), "gosimov-chat-work"), "Base directory for per-session tool workspaces")
	flag.Parse()

	cfg.provider = providerKind(strings.ToLower(strings.TrimSpace(provider)))
	cfg.apiKey = strings.TrimSpace(cfg.apiKey)
	cfg.authFile = strings.TrimSpace(cfg.authFile)
	cfg.modelID = strings.TrimSpace(cfg.modelID)

	if !isSupportedProvider(cfg.provider) {
		return config{}, fmt.Errorf("unsupported --provider %q (allowed: zen, opencode-go, openai, codex)", provider)
	}

	if cfg.modelID == "" {
		cfg.modelID = defaultModelByProvider[cfg.provider]
	}

	if cfg.apiKey == "" && cfg.authFile == "" {
		if cfg.provider.usesOAuth() {
			cfg.authFile = defaultOAuthAuthFile
		} else {
			return config{}, fmt.Errorf("either --api-key or --auth-file is required")
		}
	}

	if cfg.authFile != "" && !cfg.provider.usesOAuth() {
		return config{}, fmt.Errorf("--auth-file is only supported by codex")
	}

	if cfg.maxIterations <= 0 {
		return config{}, fmt.Errorf("--max-iterations must be greater than 0")
	}

	if cfg.maxHistory < 0 {
		return config{}, fmt.Errorf("--max-history-messages must be >= 0")
	}

	if cfg.keepRecentTok <= 0 {
		return config{}, fmt.Errorf("--compaction-keep-recent-tokens must be > 0")
	}

	if strings.TrimSpace(cfg.summaryModel) == "" {
		cfg.summaryModel = cfg.modelID
	}

	if err := os.MkdirAll(cfg.storeDir, 0o755); err != nil {
		return config{}, fmt.Errorf("creating store dir: %w", err)
	}

	if err := os.MkdirAll(cfg.workDir, 0o755); err != nil {
		return config{}, fmt.Errorf("creating work dir: %w", err)
	}

	return cfg, nil
}

func buildAuth(cfg config) (string, openai.TokenSource, string, error) {
	if cfg.authFile == "" {
		return cfg.apiKey, nil, "api key", nil
	}

	switch cfg.provider {
	case providerCodex:
		store, err := openai.NewFileCredentialsStore(cfg.authFile)
		if err != nil {
			return "", nil, "", fmt.Errorf("creating oauth credentials store: %w", err)
		}

		ts, err := openai.NewOAuthTokenSource(openai.OAuthTokenSourceConfig{
			ClientID:         openAIClientID,
			AuthorizationURL: openAIAuthURL,
			TokenURL:         openAITokenURL,
			RedirectURL:      openAIRedirectURL,
			Scopes:           parseScopes(openAIScopes),
			AuthParams: map[string]string{
				"id_token_add_organizations": "true",
				"codex_cli_simplified_flow":  "true",
				"originator":                 "pi",
			},
			Store:    store,
			StoreKey: openAIOAuthStoreKey,
		})
		if err != nil {
			return "", nil, "", fmt.Errorf("creating oauth token source: %w", err)
		}

		status, err := ensureOAuthCredentials(context.Background(), ts, store, openAIOAuthStoreKey)
		if err != nil {
			return "", nil, "", err
		}

		return "", ts, status, nil

	default:
		return "", nil, "", fmt.Errorf("--auth-file requires codex provider")
	}
}

type oauthTokenSource interface {
	AuthorizationRequest(state string) (*openai.AuthorizationRequest, error)
	ExchangeAuthorizationCode(ctx context.Context, code string, codeVerifier string) (*openai.OAuthCredentials, error)
}

type oauthCredentialsStore interface {
	Load(ctx context.Context, key string) (*openai.OAuthCredentials, error)
}

func ensureOAuthCredentials(ctx context.Context, ts oauthTokenSource, store oauthCredentialsStore, storeKey string) (string, error) {
	creds, err := store.Load(ctx, storeKey)
	if err != nil {
		return "", fmt.Errorf("loading stored credentials: %w", err)
	}

	if creds != nil && strings.TrimSpace(creds.AccessToken) != "" {
		return "reused stored credentials", nil
	}

	authReq, err := ts.AuthorizationRequest("")
	if err != nil {
		return "", fmt.Errorf("creating authorization request: %w", err)
	}

	fmt.Println("No stored OAuth credentials found for chat. Complete login:")
	fmt.Printf("1) Open this URL in your browser:\n\n%s\n\n", authReq.URL)
	fmt.Println("2) Paste the full redirect URL or the raw code and press Enter:")

	input, err := readLine()
	if err != nil {
		return "", fmt.Errorf("reading authorization input: %w", err)
	}

	code, state := parseAuthorizationInput(input)
	if state != "" && state != authReq.State {
		return "", fmt.Errorf("oauth state mismatch")
	}
	if strings.TrimSpace(code) == "" {
		return "", fmt.Errorf("missing authorization code")
	}

	if _, err := ts.ExchangeAuthorizationCode(ctx, code, authReq.CodeVerifier); err != nil {
		return "", fmt.Errorf("exchanging authorization code: %w", err)
	}

	return "completed new OAuth exchange", nil
}

func parseScopes(raw string) []string {
	raw = strings.ReplaceAll(raw, ",", " ")
	parts := strings.Fields(raw)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			out = append(out, s)
		}
	}

	return out
}

func readLine() (string, error) {
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil {
		if err == io.EOF {
			return strings.TrimSpace(line), nil
		}
		return "", err
	}

	return strings.TrimSpace(line), nil
}

func parseAuthorizationInput(input string) (code string, state string) {
	v := strings.TrimSpace(input)
	if v == "" {
		return "", ""
	}

	if u, err := url.Parse(v); err == nil && u.Scheme != "" && u.Host != "" {
		return u.Query().Get("code"), u.Query().Get("state")
	}

	if strings.Contains(v, "#") {
		parts := strings.SplitN(v, "#", 2)
		if len(parts) == 2 {
			return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		}
	}

	if strings.Contains(v, "code=") {
		q, err := url.ParseQuery(v)
		if err == nil {
			return q.Get("code"), q.Get("state")
		}
	}

	return v, ""
}

func isSupportedProvider(p providerKind) bool {
	_, ok := defaultModelByProvider[p]
	return ok
}

func (p providerKind) usesOAuth() bool {
	return p == providerCodex
}
