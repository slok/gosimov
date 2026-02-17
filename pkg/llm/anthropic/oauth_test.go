package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/slok/gosimov/pkg/llm/anthropic"
)

func TestFileCredentialsStoreRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "auth.json")

	store, err := anthropic.NewFileCredentialsStore(path)
	if err != nil {
		t.Fatalf("unexpected error creating store: %v", err)
	}

	creds := anthropic.OAuthCredentials{
		AccessToken:  "a1",
		RefreshToken: "r1",
		Expiry:       time.Now().Add(1 * time.Hour).UTC().Truncate(time.Second),
	}

	if err := store.Save(context.Background(), "anthropic", creds); err != nil {
		t.Fatalf("unexpected save error: %v", err)
	}

	got, err := store.Load(context.Background(), "anthropic")
	if err != nil {
		t.Fatalf("unexpected load error: %v", err)
	}
	if got == nil {
		t.Fatal("expected credentials, got nil")
	}

	if got.AccessToken != creds.AccessToken || got.RefreshToken != creds.RefreshToken || !got.Expiry.Equal(creds.Expiry) {
		t.Fatalf("credentials mismatch: got=%+v want=%+v", *got, creds)
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat auth file: %v", err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("expected mode 0600, got %o", fi.Mode().Perm())
	}
}

func TestOAuthTokenSourceRefresh(t *testing.T) {
	refreshCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		_ = r.ParseForm()

		if r.Form.Get("grant_type") != "refresh_token" {
			t.Fatalf("expected refresh grant, got %q", r.Form.Get("grant_type"))
		}
		if r.Form.Get("client_id") != "client-1" {
			t.Fatalf("expected client_id client-1, got %q", r.Form.Get("client_id"))
		}
		if r.Form.Get("refresh_token") != "refresh-1" {
			t.Fatalf("expected refresh_token refresh-1, got %q", r.Form.Get("refresh_token"))
		}

		refreshCalls++
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "new-access",
			"refresh_token": "new-refresh",
			"expires_in":    3600,
		})
	}))
	defer server.Close()

	store := newMemCredsStore(&anthropic.OAuthCredentials{
		AccessToken:  "expired-access",
		RefreshToken: "refresh-1",
		Expiry:       time.Now().Add(-1 * time.Minute),
	})

	ts, err := anthropic.NewOAuthTokenSource(anthropic.OAuthTokenSourceConfig{
		ClientID:         "client-1",
		AuthorizationURL: "https://auth.example.com/authorize",
		TokenURL:         server.URL,
		RedirectURL:      "http://localhost/callback",
		Store:            store,
		HTTPClient:       server.Client(),
	})
	if err != nil {
		t.Fatalf("unexpected token source error: %v", err)
	}

	token, err := ts.Token(context.Background())
	if err != nil {
		t.Fatalf("unexpected token error: %v", err)
	}
	if token != "new-access" {
		t.Fatalf("expected new-access, got %q", token)
	}
	if refreshCalls != 1 {
		t.Fatalf("expected 1 refresh call, got %d", refreshCalls)
	}

	if store.saved == nil || store.saved.AccessToken != "new-access" || store.saved.RefreshToken != "new-refresh" {
		t.Fatalf("expected saved refreshed credentials, got %+v", store.saved)
	}
}

func TestOAuthAuthorizationRequestAndExchange(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		if r.Form.Get("grant_type") != "authorization_code" {
			t.Fatalf("expected authorization_code grant, got %q", r.Form.Get("grant_type"))
		}
		if r.Form.Get("code") != "abc-code" {
			t.Fatalf("expected code abc-code, got %q", r.Form.Get("code"))
		}
		if r.Form.Get("code_verifier") == "" {
			t.Fatal("expected code_verifier")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "exchanged-access",
			"refresh_token": "exchanged-refresh",
			"expires_in":    1800,
		})
	}))
	defer server.Close()

	store := newMemCredsStore(nil)

	ts, err := anthropic.NewOAuthTokenSource(anthropic.OAuthTokenSourceConfig{
		ClientID:         "client-1",
		AuthorizationURL: "https://auth.example.com/authorize",
		TokenURL:         server.URL,
		RedirectURL:      "http://localhost/callback",
		Scopes:           []string{"openid", "offline_access"},
		Store:            store,
		StoreKey:         "anthropic",
		HTTPClient:       server.Client(),
	})
	if err != nil {
		t.Fatalf("unexpected token source error: %v", err)
	}

	authReq, err := ts.AuthorizationRequest("state-1")
	if err != nil {
		t.Fatalf("authorization request error: %v", err)
	}
	if !strings.Contains(authReq.URL, "response_type=code") {
		t.Fatalf("expected response_type=code on URL: %s", authReq.URL)
	}
	if !strings.Contains(authReq.URL, "state=state-1") {
		t.Fatalf("expected state on URL: %s", authReq.URL)
	}
	if authReq.CodeVerifier == "" {
		t.Fatal("expected code verifier")
	}

	creds, err := ts.ExchangeAuthorizationCode(context.Background(), "abc-code", authReq.CodeVerifier)
	if err != nil {
		t.Fatalf("exchange code error: %v", err)
	}
	if creds.AccessToken != "exchanged-access" {
		t.Fatalf("expected exchanged-access, got %q", creds.AccessToken)
	}
	if store.saved == nil || store.saved.AccessToken != "exchanged-access" {
		t.Fatalf("expected credentials persisted, got %+v", store.saved)
	}
}

type memCredsStore struct {
	loaded *anthropic.OAuthCredentials
	saved  *anthropic.OAuthCredentials
}

func newMemCredsStore(loaded *anthropic.OAuthCredentials) *memCredsStore {
	return &memCredsStore{loaded: loaded}
}

func (m *memCredsStore) Load(_ context.Context, _ string) (*anthropic.OAuthCredentials, error) {
	if m.loaded == nil {
		return nil, nil
	}
	c := *m.loaded
	return &c, nil
}

func (m *memCredsStore) Save(_ context.Context, _ string, creds anthropic.OAuthCredentials) error {
	c := creds
	m.saved = &c
	m.loaded = &c
	return nil
}
