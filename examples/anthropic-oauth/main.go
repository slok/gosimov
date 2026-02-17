// Example anthropic-oauth demonstrates OAuth authentication with the
// Claude provider (Anthropic Pro/Max) using authorization code + PKCE.
//
// It persists credentials in a local auth file so subsequent runs can skip
// the browser auth dance and refresh tokens automatically.
//
// Usage:
//
//	go run ./examples/anthropic-oauth
//	go run ./examples/anthropic-oauth --auth-file /tmp/anthropic-oauth-auth.json --model claude-sonnet-4-6
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"strings"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/llm/anthropic"
	"github.com/slok/gosimov/pkg/model"
)

const (
	defaultClientID         = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	defaultAuthorizationURL = "https://claude.ai/oauth/authorize"
	defaultTokenURL         = "https://console.anthropic.com/v1/oauth/token"
	defaultRedirectURL      = "https://console.anthropic.com/oauth/code/callback"
	defaultScopes           = "org:create_api_key user:profile user:inference"
	defaultAuthFile         = "/tmp/anthropic-oauth-auth.json"
	defaultStoreKey         = "anthropic"
	defaultModel            = "claude-sonnet-4-6"
)

type config struct {
	authFile         string
	storeKey         string
	clientID         string
	authorizationURL string
	tokenURL         string
	redirectURL      string
	scopes           string
	modelID          string
	prompt           string
}

func loadConfig() (config, error) {
	var cfg config
	flag.StringVar(&cfg.authFile, "auth-file", defaultAuthFile, "Path to OAuth credentials file")
	flag.StringVar(&cfg.storeKey, "store-key", defaultStoreKey, "Credentials key in the auth file")
	flag.StringVar(&cfg.clientID, "client-id", defaultClientID, "OAuth client ID")
	flag.StringVar(&cfg.authorizationURL, "authorization-url", defaultAuthorizationURL, "OAuth authorization endpoint URL")
	flag.StringVar(&cfg.tokenURL, "token-url", defaultTokenURL, "OAuth token endpoint URL")
	flag.StringVar(&cfg.redirectURL, "redirect-url", defaultRedirectURL, "OAuth redirect URL")
	flag.StringVar(&cfg.scopes, "scopes", defaultScopes, "OAuth scopes (space or comma separated)")
	flag.StringVar(&cfg.modelID, "model", defaultModel, "LLM model ID")
	flag.StringVar(&cfg.prompt, "prompt", "Say hello and confirm Claude OAuth auth works.", "User prompt")
	flag.Parse()

	if strings.TrimSpace(cfg.authFile) == "" {
		return config{}, fmt.Errorf("--auth-file is required")
	}

	if strings.TrimSpace(cfg.storeKey) == "" {
		return config{}, fmt.Errorf("--store-key is required")
	}

	if strings.TrimSpace(cfg.clientID) == "" {
		return config{}, fmt.Errorf("--client-id is required")
	}

	if strings.TrimSpace(cfg.authorizationURL) == "" {
		return config{}, fmt.Errorf("--authorization-url is required")
	}

	if strings.TrimSpace(cfg.tokenURL) == "" {
		return config{}, fmt.Errorf("--token-url is required")
	}

	if strings.TrimSpace(cfg.redirectURL) == "" {
		return config{}, fmt.Errorf("--redirect-url is required")
	}

	if strings.TrimSpace(cfg.modelID) == "" {
		return config{}, fmt.Errorf("--model is required")
	}

	if strings.TrimSpace(cfg.prompt) == "" {
		return config{}, fmt.Errorf("--prompt is required")
	}

	if len(parseScopes(cfg.scopes)) == 0 {
		return config{}, fmt.Errorf("--scopes must contain at least one scope")
	}

	return cfg, nil
}

func run(ctx context.Context) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	store, err := anthropic.NewFileCredentialsStore(cfg.authFile)
	if err != nil {
		return fmt.Errorf("creating credentials store: %w", err)
	}

	tokenSource, err := anthropic.NewClaudeOAuthTokenSource(anthropic.ClaudeOAuthTokenSourceConfig{
		ClientID:         cfg.clientID,
		AuthorizationURL: cfg.authorizationURL,
		TokenURL:         cfg.tokenURL,
		RedirectURL:      cfg.redirectURL,
		Scopes:           parseScopes(cfg.scopes),
		Store:            store,
		StoreKey:         cfg.storeKey,
	})
	if err != nil {
		return fmt.Errorf("creating oauth token source: %w", err)
	}

	authMode, err := ensureOAuthCredentials(ctx, tokenSource, store, cfg.storeKey)
	if err != nil {
		return err
	}
	fmt.Printf("Auth status: %s\n", authMode)

	provider, err := anthropic.NewClaude(anthropic.ClaudeConfig{
		TokenSource: tokenSource,
		Model:       cfg.modelID,
	})
	if err != nil {
		return fmt.Errorf("creating provider: %w", err)
	}

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:     provider,
		SystemPrompt: "You are a concise assistant.",
	})
	if err != nil {
		return fmt.Errorf("creating session: %w", err)
	}

	fmt.Printf("Auth file: %s\n", cfg.authFile)
	fmt.Printf("Model:     %s\n", cfg.modelID)
	fmt.Printf("Session:   %s\n\n", session.Session().ID)

	res, err := session.Prompt(ctx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: cfg.prompt}})
	if err != nil {
		if hint := quotaHint(err); hint != "" {
			return fmt.Errorf("running prompt: %w\n\n%s", err, hint)
		}
		return fmt.Errorf("running prompt: %w", err)
	}

	answer := "(empty response)"
	if len(res.Message.Content) > 0 {
		answer = strings.TrimSpace(res.Message.Content[0].Text)
		if answer == "" {
			answer = "(empty response)"
		}
	}

	fmt.Printf("User: %s\n\n", cfg.prompt)
	fmt.Printf("Assistant: %s\n", answer)
	if res.Message.Metadata != nil {
		fmt.Printf("Stop reason: %s\n", res.Message.Metadata.StopReason)
		fmt.Printf("Model used:  %s\n", res.Message.Metadata.Model)
	}

	return nil
}

func ensureOAuthCredentials(ctx context.Context, ts *anthropic.ClaudeOAuthTokenSource, store anthropic.OAuthCredentialsStore, storeKey string) (string, error) {
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

	fmt.Println("No stored credentials found. Complete OAuth login:")
	fmt.Printf("1) Open this URL in your browser:\n\n%s\n\n", authReq.URL)
	fmt.Println("2) After authorization, paste the full redirect URL or the raw code here and press Enter:")

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
		if errors.Is(err, io.EOF) {
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

func quotaHint(err error) string {
	msg := strings.ToLower(err.Error())
	if !strings.Contains(msg, "429") && !strings.Contains(msg, "rate limit") {
		return ""
	}

	return strings.Join([]string{
		"OAuth authentication succeeded, but the Claude backend request was rate/plan limited.",
		"If your Anthropic plan usage is exhausted, 429 responses are expected until quota resets.",
	}, "\n")
}

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}
