package auth

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/slok/gosimov/pkg/pkgerrors"
)

const (
	defaultOAuthStoreKey = "oauth"
	defaultRefreshSkew   = 30 * time.Second
	defaultTokenEncoding = OAuthTokenRequestEncodingForm
)

// OAuthTokenRequestEncoding configures token endpoint payload encoding.
type OAuthTokenRequestEncoding string

const (
	// OAuthTokenRequestEncodingForm uses application/x-www-form-urlencoded.
	OAuthTokenRequestEncodingForm OAuthTokenRequestEncoding = "form"
	// OAuthTokenRequestEncodingJSON uses application/json.
	OAuthTokenRequestEncodingJSON OAuthTokenRequestEncoding = "json"
)

// OAuthAuthorizationStateMode configures how OAuth state values are generated.
type OAuthAuthorizationStateMode string

const (
	// OAuthAuthorizationStateModeRandom uses a random state by default.
	OAuthAuthorizationStateModeRandom OAuthAuthorizationStateMode = "random"
	// OAuthAuthorizationStateModeCodeVerifier uses the PKCE verifier as state.
	OAuthAuthorizationStateModeCodeVerifier OAuthAuthorizationStateMode = "code_verifier"
)

// OAuthCredentials are persisted credentials for OAuth authentication.
type OAuthCredentials struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	Expiry       time.Time `json:"expiry"`
}

// OAuthCredentialsStore stores and loads OAuth credentials.
type OAuthCredentialsStore interface {
	Load(ctx context.Context, key string) (*OAuthCredentials, error)
	Save(ctx context.Context, key string, creds OAuthCredentials) error
}

// FileCredentialsStore stores OAuth credentials in a JSON file.
type FileCredentialsStore struct {
	path string
}

// NewFileCredentialsStore creates a file-backed credentials store.
func NewFileCredentialsStore(path string) (*FileCredentialsStore, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("path is required: %w", pkgerrors.ErrNotValid)
	}

	return &FileCredentialsStore{path: path}, nil
}

func (s *FileCredentialsStore) Load(_ context.Context, key string) (*OAuthCredentials, error) {
	if strings.TrimSpace(key) == "" {
		return nil, fmt.Errorf("store key is required: %w", pkgerrors.ErrNotValid)
	}

	b, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}

		return nil, fmt.Errorf("reading auth file: %w", err)
	}

	data := map[string]OAuthCredentials{}
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, fmt.Errorf("unmarshaling auth file: %w", err)
	}

	creds, ok := data[key]
	if !ok {
		return nil, nil
	}

	return &creds, nil
}

func (s *FileCredentialsStore) Save(_ context.Context, key string, creds OAuthCredentials) error {
	if strings.TrimSpace(key) == "" {
		return fmt.Errorf("store key is required: %w", pkgerrors.ErrNotValid)
	}

	data := map[string]OAuthCredentials{}
	b, err := os.ReadFile(s.path)
	if err == nil {
		if err := json.Unmarshal(b, &data); err != nil {
			return fmt.Errorf("unmarshaling auth file: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("reading auth file: %w", err)
	}

	data[key] = creds

	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return fmt.Errorf("creating auth dir: %w", err)
	}

	b, err = json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling auth file: %w", err)
	}

	if err := os.WriteFile(s.path, b, 0o600); err != nil {
		return fmt.Errorf("writing auth file: %w", err)
	}

	return nil
}

// OAuthTokenSourceConfig configures OAuth bearer token retrieval.
type OAuthTokenSourceConfig struct {
	ClientID         string
	AuthorizationURL string
	TokenURL         string
	RedirectURL      string
	Scopes           []string
	AuthParams       map[string]string
	AuthCodeParams   map[string]string
	RefreshParams    map[string]string
	StateMode        OAuthAuthorizationStateMode
	TokenEncoding    OAuthTokenRequestEncoding
	Store            OAuthCredentialsStore
	StoreKey         string
	RefreshSkew      time.Duration
	HTTPClient       *http.Client
}

func (c *OAuthTokenSourceConfig) defaults() error {
	if strings.TrimSpace(c.ClientID) == "" {
		return fmt.Errorf("client id is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.AuthorizationURL) == "" {
		return fmt.Errorf("authorization URL is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.TokenURL) == "" {
		return fmt.Errorf("token URL is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.RedirectURL) == "" {
		return fmt.Errorf("redirect URL is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Store == nil {
		return fmt.Errorf("credentials store is required: %w", pkgerrors.ErrNotValid)
	}

	if c.StateMode == "" {
		c.StateMode = OAuthAuthorizationStateModeRandom
	}
	if c.StateMode != OAuthAuthorizationStateModeRandom && c.StateMode != OAuthAuthorizationStateModeCodeVerifier {
		return fmt.Errorf("unsupported oauth state mode %q: %w", c.StateMode, pkgerrors.ErrNotValid)
	}

	if c.TokenEncoding == "" {
		c.TokenEncoding = defaultTokenEncoding
	}
	if c.TokenEncoding != OAuthTokenRequestEncodingForm && c.TokenEncoding != OAuthTokenRequestEncodingJSON {
		return fmt.Errorf("unsupported oauth token encoding %q: %w", c.TokenEncoding, pkgerrors.ErrNotValid)
	}

	if c.StoreKey == "" {
		c.StoreKey = defaultOAuthStoreKey
	}

	if c.RefreshSkew <= 0 {
		c.RefreshSkew = defaultRefreshSkew
	}

	if c.HTTPClient == nil {
		c.HTTPClient = http.DefaultClient
	}

	return nil
}

// OAuthTokenSource provides OAuth access tokens, refreshing and persisting as needed.
type OAuthTokenSource struct {
	clientID         string
	authorizationURL string
	tokenURL         string
	redirectURL      string
	scopes           []string
	authParams       map[string]string
	authCodeParams   map[string]string
	refreshParams    map[string]string
	stateMode        OAuthAuthorizationStateMode
	tokenEncoding    OAuthTokenRequestEncoding
	store            OAuthCredentialsStore
	storeKey         string
	refreshSkew      time.Duration
	httpClient       *http.Client

	mu          sync.Mutex
	credentials *OAuthCredentials
}

// NewOAuthTokenSource creates an OAuth token source.
func NewOAuthTokenSource(cfg OAuthTokenSourceConfig) (*OAuthTokenSource, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid oauth token source config: %w", err)
	}

	return &OAuthTokenSource{
		clientID:         cfg.ClientID,
		authorizationURL: cfg.AuthorizationURL,
		tokenURL:         cfg.TokenURL,
		redirectURL:      cfg.RedirectURL,
		scopes:           append([]string(nil), cfg.Scopes...),
		authParams:       cloneMap(cfg.AuthParams),
		authCodeParams:   cloneMap(cfg.AuthCodeParams),
		refreshParams:    cloneMap(cfg.RefreshParams),
		stateMode:        cfg.StateMode,
		tokenEncoding:    cfg.TokenEncoding,
		store:            cfg.Store,
		storeKey:         cfg.StoreKey,
		refreshSkew:      cfg.RefreshSkew,
		httpClient:       cfg.HTTPClient,
	}, nil
}

// AuthorizationRequest describes the browser authorization request.
type AuthorizationRequest struct {
	URL          string
	State        string
	CodeVerifier string
}

// AuthorizationRequest returns an OAuth authorization URL and PKCE verifier.
func (s *OAuthTokenSource) AuthorizationRequest(state string) (*AuthorizationRequest, error) {
	if state == "" {
		generated, err := randomURLSafe(16)
		if err != nil {
			return nil, fmt.Errorf("generating state: %w", err)
		}
		state = generated
	}

	codeVerifier, err := randomURLSafe(32)
	if err != nil {
		return nil, fmt.Errorf("generating code verifier: %w", err)
	}

	challenge := sha256.Sum256([]byte(codeVerifier))
	codeChallenge := base64.RawURLEncoding.EncodeToString(challenge[:])
	if s.stateMode == OAuthAuthorizationStateModeCodeVerifier {
		state = codeVerifier
	}

	u, err := url.Parse(s.authorizationURL)
	if err != nil {
		return nil, fmt.Errorf("parsing authorization URL: %w", err)
	}

	q := u.Query()
	q.Set("response_type", "code")
	q.Set("client_id", s.clientID)
	q.Set("redirect_uri", s.redirectURL)
	q.Set("code_challenge", codeChallenge)
	q.Set("code_challenge_method", "S256")
	q.Set("state", state)
	if len(s.scopes) > 0 {
		q.Set("scope", strings.Join(s.scopes, " "))
	}
	for k, v := range s.authParams {
		q.Set(k, v)
	}
	u.RawQuery = q.Encode()

	return &AuthorizationRequest{
		URL:          u.String(),
		State:        state,
		CodeVerifier: codeVerifier,
	}, nil
}

// ExchangeAuthorizationCode exchanges the OAuth code and persists credentials.
func (s *OAuthTokenSource) ExchangeAuthorizationCode(ctx context.Context, code string, codeVerifier string) (*OAuthCredentials, error) {
	if strings.TrimSpace(code) == "" {
		return nil, fmt.Errorf("code is required: %w", pkgerrors.ErrNotValid)
	}
	if strings.TrimSpace(codeVerifier) == "" {
		return nil, fmt.Errorf("code verifier is required: %w", pkgerrors.ErrNotValid)
	}

	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("client_id", s.clientID)
	form.Set("code", code)
	form.Set("code_verifier", codeVerifier)
	form.Set("redirect_uri", s.redirectURL)
	if s.stateMode == OAuthAuthorizationStateModeCodeVerifier {
		form.Set("state", codeVerifier)
	}
	for k, v := range s.authCodeParams {
		form.Set(k, v)
	}

	creds, err := s.exchangeTokenRequest(ctx, form)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.store.Save(ctx, s.storeKey, *creds); err != nil {
		return nil, fmt.Errorf("saving oauth credentials: %w", err)
	}

	s.credentials = creds

	return creds, nil
}

// Token returns a bearer token, refreshing stored credentials if needed.
func (s *OAuthTokenSource) Token(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.ensureCredentialsLoaded(ctx); err != nil {
		return "", err
	}

	if s.credentials == nil {
		return "", fmt.Errorf("oauth credentials not initialized, complete login flow first: %w", pkgerrors.ErrNotValid)
	}

	if s.credentials.AccessToken != "" && time.Now().Add(s.refreshSkew).Before(s.credentials.Expiry) {
		return s.credentials.AccessToken, nil
	}

	if s.credentials.RefreshToken == "" {
		return "", fmt.Errorf("oauth token expired and refresh token is missing: %w", pkgerrors.ErrNotValid)
	}

	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("client_id", s.clientID)
	form.Set("refresh_token", s.credentials.RefreshToken)
	for k, v := range s.refreshParams {
		form.Set(k, v)
	}

	creds, err := s.exchangeTokenRequest(ctx, form)
	if err != nil {
		return "", err
	}

	if creds.RefreshToken == "" {
		creds.RefreshToken = s.credentials.RefreshToken
	}

	if err := s.store.Save(ctx, s.storeKey, *creds); err != nil {
		return "", fmt.Errorf("saving refreshed oauth credentials: %w", err)
	}

	s.credentials = creds

	return s.credentials.AccessToken, nil
}

func (s *OAuthTokenSource) ensureCredentialsLoaded(ctx context.Context) error {
	if s.credentials != nil {
		return nil
	}

	creds, err := s.store.Load(ctx, s.storeKey)
	if err != nil {
		return fmt.Errorf("loading oauth credentials: %w", err)
	}

	s.credentials = creds

	return nil
}

func (s *OAuthTokenSource) exchangeTokenRequest(ctx context.Context, form url.Values) (*OAuthCredentials, error) {
	var (
		requestBody io.Reader
		contentType string
	)

	switch s.tokenEncoding {
	case OAuthTokenRequestEncodingJSON:
		payload := map[string]string{}
		for k, values := range form {
			if len(values) > 0 {
				payload[k] = values[0]
			}
		}

		b, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshaling token request: %w", err)
		}
		requestBody = bytes.NewReader(b)
		contentType = "application/json"

	default:
		requestBody = strings.NewReader(form.Encode())
		contentType = "application/x-www-form-urlencoded"
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.tokenURL, requestBody)
	if err != nil {
		return nil, fmt.Errorf("creating token request: %w", err)
	}
	httpReq.Header.Set("Content-Type", contentType)

	httpResp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing token request: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading token response body: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, parseOAuthError(httpResp.StatusCode, respBody)
	}

	var tr oauthTokenResponse
	if err := json.Unmarshal(respBody, &tr); err != nil {
		return nil, fmt.Errorf("unmarshaling token response: %w", err)
	}

	if strings.TrimSpace(tr.AccessToken) == "" {
		return nil, fmt.Errorf("missing access token in token response: %w", pkgerrors.ErrLLMError)
	}

	if tr.ExpiresIn <= 0 {
		tr.ExpiresIn = 3600
	}

	return &OAuthCredentials{
		AccessToken:  tr.AccessToken,
		RefreshToken: tr.RefreshToken,
		Expiry:       time.Now().Add(time.Duration(tr.ExpiresIn) * time.Second),
	}, nil
}

type oauthTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	TokenType    string `json:"token_type"`
}

func parseOAuthError(statusCode int, body []byte) error {
	var resp struct {
		Error            string `json:"error"`
		ErrorDescription string `json:"error_description"`
	}

	if err := json.Unmarshal(body, &resp); err == nil {
		detail := strings.TrimSpace(resp.ErrorDescription)
		if detail == "" {
			detail = strings.TrimSpace(resp.Error)
		}
		if detail == "" {
			detail = strings.TrimSpace(string(body))
		}

		return fmt.Errorf("oauth token endpoint error (status %d): %s: %w", statusCode, detail, pkgerrors.ErrLLMError)
	}

	return fmt.Errorf("oauth token endpoint error (status %d): %s: %w", statusCode, string(body), pkgerrors.ErrLLMError)
}

func randomURLSafe(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("random bytes: %w", err)
	}

	return base64.RawURLEncoding.EncodeToString(b), nil
}

func cloneMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}

	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}

	return out
}
