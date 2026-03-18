package opencodego

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/internal/anthropicmsg"
	"github.com/slok/gosimov/pkg/llm/internal/openaichat"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

const (
	defaultBaseURL = "https://opencode.ai/zen/go/v1"
	providerID     = "opencode-go"
)

// Config configures the OpenCode Go provider.
type Config struct {
	TokenSource TokenSource
	BaseURL     string
	Model       string
	Client      *http.Client
}

func (c *Config) defaults() (model.LLMModelInfo, modelAPIFormat, error) {
	if c.TokenSource == nil {
		return model.LLMModelInfo{}, "", fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.Model) == "" {
		return model.LLMModelInfo{}, "", fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.BaseURL) == "" {
		c.BaseURL = defaultBaseURL
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	info, ok := ModelByID(c.Model)
	if !ok {
		return model.LLMModelInfo{}, "", fmt.Errorf("unsupported opencode-go model %q: %w", c.Model, pkgerrors.ErrNotValid)
	}

	format, ok := ModelFormatByID(c.Model)
	if !ok {
		return model.LLMModelInfo{}, "", fmt.Errorf("missing api format for opencode-go model %q: %w", c.Model, pkgerrors.ErrNotValid)
	}

	return info, format, nil
}

// New creates a new OpenCode Go provider.
func New(cfg Config) (llm.Provider, error) {
	info, format, err := cfg.defaults()
	if err != nil {
		return nil, fmt.Errorf("invalid opencode-go provider config: %w", err)
	}

	switch format {
	case modelAPIFormatOpenAICompatible:
		return openaichat.New(openaichat.Config{
			TokenSource: cfg.TokenSource,
			BaseURL:     cfg.BaseURL,
			Model:       cfg.Model,
			ModelInfo:   info,
			ProviderID:  providerID,
			Client:      cfg.Client,
		})

	case modelAPIFormatAnthropic:
		return anthropicmsg.New(anthropicmsg.Config{
			TokenSource: cfg.TokenSource,
			BaseURL:     cfg.BaseURL,
			Model:       cfg.Model,
			ModelInfo:   info,
			Client:      cfg.Client,
			Options: anthropicmsg.Options{
				ProviderID:       providerID,
				AuthMode:         anthropicmsg.AuthModeAPIKey,
				DefaultMaxTokens: info.MaxOutputTokens,
			},
		})

	default:
		return nil, fmt.Errorf("unsupported api format %q for opencode-go model %q: %w", format, cfg.Model, pkgerrors.ErrNotValid)
	}
}
