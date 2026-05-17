package customopenaicompatible

import llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"

// TokenSource returns bearer tokens used to call OpenAI-compatible APIs.
type TokenSource = llmauth.TokenSource

// APIKeyTokenSource is a static [TokenSource] backed by a single API key.
type APIKeyTokenSource = llmauth.APIKeyTokenSource

// NewAPIKeyTokenSource creates a static token source from an API key.
func NewAPIKeyTokenSource(apiKey string) TokenSource {
	return llmauth.NewAPIKeyTokenSource(apiKey)
}
