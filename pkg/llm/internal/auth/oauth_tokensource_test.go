package auth_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"
)

func TestOAuthTokenSourceJSONEncodingAndCodeVerifierState(t *testing.T) {
	refreshCalled := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Fatalf("expected application/json content type, got %q", got)
		}

		var payload map[string]string
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode json body: %v", err)
		}

		switch payload["grant_type"] {
		case "authorization_code":
			if payload["state"] == "" || payload["state"] != payload["code_verifier"] {
				t.Fatalf("expected state to match code_verifier, got state=%q verifier=%q", payload["state"], payload["code_verifier"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "access-1",
				"refresh_token": "refresh-1",
				"expires_in":    1,
			})

		case "refresh_token":
			refreshCalled = true
			if payload["refresh_token"] != "refresh-1" {
				t.Fatalf("expected refresh token refresh-1, got %q", payload["refresh_token"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "access-2",
				"refresh_token": "refresh-2",
				"expires_in":    3600,
			})

		default:
			t.Fatalf("unexpected grant_type: %q", payload["grant_type"])
		}
	}))
	defer server.Close()

	store := &memCredsStore{}
	ts, err := llmauth.NewOAuthTokenSource(llmauth.OAuthTokenSourceConfig{
		ClientID:         "client-1",
		AuthorizationURL: "https://auth.example.com/authorize",
		TokenURL:         server.URL,
		RedirectURL:      "https://app.example.com/callback",
		Scopes:           []string{"s1", "s2"},
		AuthParams:       map[string]string{"code": "true"},
		StateMode:        llmauth.OAuthAuthorizationStateModeCodeVerifier,
		TokenEncoding:    llmauth.OAuthTokenRequestEncodingJSON,
		Store:            store,
		StoreKey:         "k",
		HTTPClient:       server.Client(),
	})
	if err != nil {
		t.Fatalf("new oauth token source: %v", err)
	}

	authReq, err := ts.AuthorizationRequest("")
	if err != nil {
		t.Fatalf("authorization request: %v", err)
	}
	if authReq.State != authReq.CodeVerifier {
		t.Fatalf("expected state to equal code verifier")
	}
	if !strings.Contains(authReq.URL, "code=true") {
		t.Fatalf("expected auth param code=true in URL")
	}

	if _, err := ts.ExchangeAuthorizationCode(context.Background(), "abc-code", authReq.CodeVerifier); err != nil {
		t.Fatalf("exchange auth code: %v", err)
	}

	store.loaded.Expiry = time.Now().Add(-1 * time.Minute)
	tok, err := ts.Token(context.Background())
	if err != nil {
		t.Fatalf("token refresh: %v", err)
	}
	if tok != "access-2" {
		t.Fatalf("expected access-2 token, got %q", tok)
	}
	if !refreshCalled {
		t.Fatalf("expected refresh call")
	}
}

type memCredsStore struct {
	loaded *llmauth.OAuthCredentials
	saved  *llmauth.OAuthCredentials
}

func (m *memCredsStore) Load(_ context.Context, _ string) (*llmauth.OAuthCredentials, error) {
	if m.loaded == nil {
		return nil, nil
	}
	c := *m.loaded
	return &c, nil
}

func (m *memCredsStore) Save(_ context.Context, _ string, creds llmauth.OAuthCredentials) error {
	c := creds
	m.saved = &c
	m.loaded = &c
	return nil
}
