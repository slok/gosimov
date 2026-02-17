package auth

import (
	"context"
	"fmt"

	"github.com/slok/gosimov/pkg/pkgerrors"
)

// TokenSource returns bearer tokens used to call LLM APIs.
type TokenSource interface {
	Token(ctx context.Context) (string, error)
}

// APIKeyTokenSource is a static [TokenSource] backed by a single API key.
type APIKeyTokenSource struct {
	apiKey string
}

// NewAPIKeyTokenSource creates a static token source from an API key.
func NewAPIKeyTokenSource(apiKey string) TokenSource {
	return APIKeyTokenSource{apiKey: apiKey}
}

func (s APIKeyTokenSource) Token(_ context.Context) (string, error) {
	if s.apiKey == "" {
		return "", fmt.Errorf("empty token: %w", pkgerrors.ErrNotValid)
	}

	return s.apiKey, nil
}
