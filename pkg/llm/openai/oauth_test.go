package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/llm/openai"
)

func TestFileCredentialsStoreRoundTrip(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Save and load should return the same credentials with correct file permissions.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				tmpDir := t.TempDir()
				path := filepath.Join(tmpDir, "auth.json")

				store, err := openai.NewFileCredentialsStore(path)
				require.NoError(err)

				creds := openai.OAuthCredentials{
					AccessToken:  "a1",
					RefreshToken: "r1",
					Expiry:       time.Now().Add(1 * time.Hour).UTC().Truncate(time.Second),
				}

				err = store.Save(context.Background(), "openai", creds)
				require.NoError(err)

				got, err := store.Load(context.Background(), "openai")
				require.NoError(err)
				require.NotNil(got)

				assert.Equal(creds.AccessToken, got.AccessToken)
				assert.Equal(creds.RefreshToken, got.RefreshToken)
				assert.True(got.Expiry.Equal(creds.Expiry))

				fi, err := os.Stat(path)
				require.NoError(err)
				assert.Equal(os.FileMode(0o600), fi.Mode().Perm())
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

func TestOAuthTokenSource(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Token refresh should call the token endpoint and persist new credentials.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				refreshCalls := 0
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					require.Equal(http.MethodPost, r.Method)
					_ = r.ParseForm()

					require.Equal("refresh_token", r.Form.Get("grant_type"))
					require.Equal("client-1", r.Form.Get("client_id"))
					require.Equal("refresh-1", r.Form.Get("refresh_token"))

					refreshCalls++
					_ = json.NewEncoder(w).Encode(map[string]any{
						"access_token":  "new-access",
						"refresh_token": "new-refresh",
						"expires_in":    3600,
					})
				}))
				defer server.Close()

				store := newMemCredsStore(&openai.OAuthCredentials{
					AccessToken:  "expired-access",
					RefreshToken: "refresh-1",
					Expiry:       time.Now().Add(-1 * time.Minute),
				})

				ts, err := openai.NewOAuthTokenSource(openai.OAuthTokenSourceConfig{
					ClientID:         "client-1",
					AuthorizationURL: "https://auth.example.com/authorize",
					TokenURL:         server.URL,
					RedirectURL:      "http://localhost/callback",
					Store:            store,
					HTTPClient:       server.Client(),
				})
				require.NoError(err)

				token, err := ts.Token(context.Background())
				require.NoError(err)
				assert.Equal("new-access", token)
				assert.Equal(1, refreshCalls)

				require.NotNil(store.saved)
				assert.Equal("new-access", store.saved.AccessToken)
				assert.Equal("new-refresh", store.saved.RefreshToken)
			},
		},

		"Authorization request and code exchange should work end-to-end.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_ = r.ParseForm()
					require.Equal("authorization_code", r.Form.Get("grant_type"))
					require.Equal("abc-code", r.Form.Get("code"))
					require.NotEmpty(r.Form.Get("code_verifier"))

					_ = json.NewEncoder(w).Encode(map[string]any{
						"access_token":  "exchanged-access",
						"refresh_token": "exchanged-refresh",
						"expires_in":    1800,
					})
				}))
				defer server.Close()

				store := newMemCredsStore(nil)

				ts, err := openai.NewOAuthTokenSource(openai.OAuthTokenSourceConfig{
					ClientID:         "client-1",
					AuthorizationURL: "https://auth.example.com/authorize",
					TokenURL:         server.URL,
					RedirectURL:      "http://localhost/callback",
					Scopes:           []string{"openid", "offline_access"},
					Store:            store,
					StoreKey:         "openai",
					HTTPClient:       server.Client(),
				})
				require.NoError(err)

				authReq, err := ts.AuthorizationRequest("state-1")
				require.NoError(err)
				assert.Contains(authReq.URL, "response_type=code")
				assert.Contains(authReq.URL, "state=state-1")
				assert.NotEmpty(authReq.CodeVerifier)

				creds, err := ts.ExchangeAuthorizationCode(context.Background(), "abc-code", authReq.CodeVerifier)
				require.NoError(err)
				assert.Equal("exchanged-access", creds.AccessToken)

				require.NotNil(store.saved)
				assert.Equal("exchanged-access", store.saved.AccessToken)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

type memCredsStore struct {
	loaded *openai.OAuthCredentials
	saved  *openai.OAuthCredentials
}

func newMemCredsStore(loaded *openai.OAuthCredentials) *memCredsStore {
	return &memCredsStore{loaded: loaded}
}

func (m *memCredsStore) Load(_ context.Context, _ string) (*openai.OAuthCredentials, error) {
	if m.loaded == nil {
		return nil, nil
	}
	c := *m.loaded
	return &c, nil
}

func (m *memCredsStore) Save(_ context.Context, _ string, creds openai.OAuthCredentials) error {
	c := creds
	m.saved = &c
	m.loaded = &c
	return nil
}
