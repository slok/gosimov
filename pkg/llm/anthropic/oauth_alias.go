package anthropic

import llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"

// OAuthCredentials are persisted credentials for OAuth authentication.
type OAuthCredentials = llmauth.OAuthCredentials

// OAuthCredentialsStore stores and loads OAuth credentials.
type OAuthCredentialsStore = llmauth.OAuthCredentialsStore

// FileCredentialsStore stores OAuth credentials in a JSON file.
type FileCredentialsStore = llmauth.FileCredentialsStore

// NewFileCredentialsStore creates a file-backed credentials store.
func NewFileCredentialsStore(path string) (*FileCredentialsStore, error) {
	return llmauth.NewFileCredentialsStore(path)
}

// OAuthTokenSourceConfig configures OAuth bearer token retrieval.
type OAuthTokenSourceConfig = llmauth.OAuthTokenSourceConfig

// OAuthTokenRequestEncoding configures token endpoint payload encoding.
type OAuthTokenRequestEncoding = llmauth.OAuthTokenRequestEncoding

const (
	OAuthTokenRequestEncodingForm = llmauth.OAuthTokenRequestEncodingForm
	OAuthTokenRequestEncodingJSON = llmauth.OAuthTokenRequestEncodingJSON
)

// OAuthAuthorizationStateMode configures how OAuth state values are generated.
type OAuthAuthorizationStateMode = llmauth.OAuthAuthorizationStateMode

const (
	OAuthAuthorizationStateModeRandom       = llmauth.OAuthAuthorizationStateModeRandom
	OAuthAuthorizationStateModeCodeVerifier = llmauth.OAuthAuthorizationStateModeCodeVerifier
)

// OAuthTokenSource provides OAuth access tokens, refreshing and persisting as needed.
type OAuthTokenSource = llmauth.OAuthTokenSource

// AuthorizationRequest describes the browser authorization request.
type AuthorizationRequest = llmauth.AuthorizationRequest

// NewOAuthTokenSource creates an OAuth token source.
func NewOAuthTokenSource(cfg OAuthTokenSourceConfig) (*OAuthTokenSource, error) {
	return llmauth.NewOAuthTokenSource(cfg)
}
