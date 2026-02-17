package anthropic

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

const anthropicProviderID = "anthropic"

// Config configures the Anthropic Messages API provider using API key authentication.
type Config struct {
	TokenSource TokenSource
	BaseURL     string
	Model       string
	Tools       []tool.Tool
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

	return newProvider(providerConfig{
		TokenSource: cfg.TokenSource,
		BaseURL:     cfg.BaseURL,
		Model:       cfg.Model,
		ModelInfo:   info,
		Tools:       convertTools(cfg.Tools, nil),
		Client:      cfg.Client,
		Options: providerOptions{
			providerID:       anthropicProviderID,
			authMode:         authModeAPIKey,
			defaultMaxTokens: info.MaxOutputTokens,
		},
	})
}
