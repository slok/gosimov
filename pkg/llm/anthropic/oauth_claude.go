package anthropic

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

const (
	defaultClaudeOAuthClientID         = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	defaultClaudeOAuthAuthorizationURL = "https://claude.ai/oauth/authorize"
	defaultClaudeOAuthTokenURL         = "https://console.anthropic.com/v1/oauth/token"
	defaultClaudeOAuthRedirectURL      = "https://console.anthropic.com/oauth/code/callback"
	defaultClaudeOAuthScopes           = "org:create_api_key user:profile user:inference"
	defaultClaudeOAuthStoreKey         = "anthropic"
)

// ClaudeOAuthTokenSourceConfig configures Claude Pro/Max OAuth bearer token retrieval.
type ClaudeOAuthTokenSourceConfig struct {
	ClientID         string
	AuthorizationURL string
	TokenURL         string
	RedirectURL      string
	Scopes           []string
	Store            OAuthCredentialsStore
	StoreKey         string
	RefreshSkew      time.Duration
	HTTPClient       *http.Client
}

// ClaudeOAuthTokenSource provides OAuth access tokens for Claude Pro/Max.
type ClaudeOAuthTokenSource = OAuthTokenSource

// NewClaudeOAuthTokenSource creates a Claude OAuth token source using shared OAuth internals.
func NewClaudeOAuthTokenSource(cfg ClaudeOAuthTokenSourceConfig) (*ClaudeOAuthTokenSource, error) {
	if cfg.Store == nil {
		return nil, fmt.Errorf("credentials store is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(cfg.ClientID) == "" {
		cfg.ClientID = defaultClaudeOAuthClientID
	}
	if strings.TrimSpace(cfg.AuthorizationURL) == "" {
		cfg.AuthorizationURL = defaultClaudeOAuthAuthorizationURL
	}
	if strings.TrimSpace(cfg.TokenURL) == "" {
		cfg.TokenURL = defaultClaudeOAuthTokenURL
	}
	if strings.TrimSpace(cfg.RedirectURL) == "" {
		cfg.RedirectURL = defaultClaudeOAuthRedirectURL
	}
	if len(cfg.Scopes) == 0 {
		cfg.Scopes = strings.Fields(defaultClaudeOAuthScopes)
	}
	if strings.TrimSpace(cfg.StoreKey) == "" {
		cfg.StoreKey = defaultClaudeOAuthStoreKey
	}

	return NewOAuthTokenSource(OAuthTokenSourceConfig{
		ClientID:         cfg.ClientID,
		AuthorizationURL: cfg.AuthorizationURL,
		TokenURL:         cfg.TokenURL,
		RedirectURL:      cfg.RedirectURL,
		Scopes:           cfg.Scopes,
		AuthParams: map[string]string{
			"code": "true",
		},
		StateMode:     llmauth.OAuthAuthorizationStateModeCodeVerifier,
		TokenEncoding: llmauth.OAuthTokenRequestEncodingJSON,
		Store:         cfg.Store,
		StoreKey:      cfg.StoreKey,
		RefreshSkew:   cfg.RefreshSkew,
		HTTPClient:    cfg.HTTPClient,
	})
}
