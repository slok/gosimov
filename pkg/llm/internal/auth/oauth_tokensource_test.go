package auth_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"
)

func TestOAuthTokenSource(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"JSON encoding and code verifier state should work for authorization exchange and token refresh.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				refreshCalled := false
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					require.Equal("application/json", r.Header.Get("Content-Type"))

					var payload map[string]string
					err := json.NewDecoder(r.Body).Decode(&payload)
					require.NoError(err)

					switch payload["grant_type"] {
					case "authorization_code":
						require.NotEmpty(payload["state"])
						require.Equal(payload["state"], payload["code_verifier"])

						_ = json.NewEncoder(w).Encode(map[string]any{
							"access_token":  "access-1",
							"refresh_token": "refresh-1",
							"expires_in":    1,
						})

					case "refresh_token":
						refreshCalled = true
						require.Equal("refresh-1", payload["refresh_token"])

						_ = json.NewEncoder(w).Encode(map[string]any{
							"access_token":  "access-2",
							"refresh_token": "refresh-2",
							"expires_in":    3600,
						})

					default:
						require.Failf("unexpected grant_type: %q", payload["grant_type"])
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
				require.NoError(err)

				authReq, err := ts.AuthorizationRequest("")
				require.NoError(err)
				assert.Equal(authReq.State, authReq.CodeVerifier)
				assert.Contains(authReq.URL, "code=true")

				_, err = ts.ExchangeAuthorizationCode(context.Background(), "abc-code", authReq.CodeVerifier)
				require.NoError(err)

				store.loaded.Expiry = time.Now().Add(-1 * time.Minute)
				tok, err := ts.Token(context.Background())
				require.NoError(err)
				assert.Equal("access-2", tok)
				assert.True(refreshCalled)
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
