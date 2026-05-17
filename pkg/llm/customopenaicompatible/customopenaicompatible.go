package customopenaicompatible

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/internal/openaichat"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

const defaultProviderID = "custom-openai-compatible"

// Config configures a custom OpenAI-compatible chat completions provider.
type Config struct {
	BaseURL         string
	Model           string
	ContextWindow   int
	MaxOutputTokens int
	APIKey          string
	TokenSource     TokenSource
	ProviderID      string
	Client          *http.Client
}

func (c *Config) defaults() (model.LLMModelInfo, error) {
	if strings.TrimSpace(c.BaseURL) == "" {
		return model.LLMModelInfo{}, fmt.Errorf("base url is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.Model) == "" {
		return model.LLMModelInfo{}, fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if c.ContextWindow <= 0 {
		return model.LLMModelInfo{}, fmt.Errorf("context window must be > 0: %w", pkgerrors.ErrNotValid)
	}

	if c.MaxOutputTokens <= 0 {
		return model.LLMModelInfo{}, fmt.Errorf("max output tokens must be > 0: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.APIKey) != "" && c.TokenSource != nil {
		return model.LLMModelInfo{}, fmt.Errorf("api key and token source are mutually exclusive: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.APIKey) != "" {
		c.TokenSource = NewAPIKeyTokenSource(c.APIKey)
	}

	if strings.TrimSpace(c.ProviderID) == "" {
		c.ProviderID = defaultProviderID
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	return model.LLMModelInfo{
		ID:              c.Model,
		Name:            c.Model,
		ContextWindow:   c.ContextWindow,
		MaxOutputTokens: c.MaxOutputTokens,
	}, nil
}

// New creates a new custom OpenAI-compatible provider.
func New(cfg Config) (llm.Provider, error) {
	info, err := cfg.defaults()
	if err != nil {
		return nil, fmt.Errorf("invalid custom openai-compatible provider config: %w", err)
	}

	return openaichat.New(openaichat.Config{
		TokenSource: cfg.TokenSource,
		BaseURL:     cfg.BaseURL,
		Model:       cfg.Model,
		ModelInfo:   info,
		ProviderID:  cfg.ProviderID,
		Client:      cfg.Client,
	})
}
