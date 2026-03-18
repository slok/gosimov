package anthropic

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/internal/anthropicmsg"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

const (
	anthropicProviderID     = "anthropic"
	defaultAnthropicBaseURL = "https://api.anthropic.com/v1"
)

// Config configures the Anthropic Messages API provider using API key authentication.
type Config struct {
	TokenSource TokenSource
	BaseURL     string
	Model       string
	Client      *http.Client
}

func (c *Config) defaults() (model.LLMModelInfo, error) {
	if c.TokenSource == nil {
		return model.LLMModelInfo{}, fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.Model) == "" {
		return model.LLMModelInfo{}, fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	info, ok := ModelByID(c.Model)
	if !ok {
		return model.LLMModelInfo{}, fmt.Errorf("unsupported anthropic model %q: %w", c.Model, pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.BaseURL) == "" {
		c.BaseURL = defaultAnthropicBaseURL
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	return info, nil
}

// NewAnthropic creates a new Anthropic Messages provider using API key authentication.
func NewAnthropic(cfg Config) (llm.Provider, error) {
	info, err := cfg.defaults()
	if err != nil {
		return nil, fmt.Errorf("invalid anthropic provider config: %w", err)
	}

	return anthropicmsg.New(anthropicmsg.Config{
		TokenSource: cfg.TokenSource,
		BaseURL:     cfg.BaseURL,
		Model:       cfg.Model,
		ModelInfo:   info,
		Client:      cfg.Client,
		Options: anthropicmsg.Options{
			ProviderID:       anthropicProviderID,
			AuthMode:         anthropicmsg.AuthModeAPIKey,
			DefaultMaxTokens: info.MaxOutputTokens,
		},
	})
}
