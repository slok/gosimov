package zen

import (
	"fmt"
	"net/http"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/internal/openaichat"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

const defaultBaseURL = "https://opencode.ai/zen/v1"

// Config configures the OpenCode Zen provider.
type Config struct {
	TokenSource TokenSource
	Model       string
	Tools       []tool.Tool
	Client      *http.Client
}

func (c *Config) defaults() (model.LLMModelInfo, error) {
	if c.TokenSource == nil {
		return model.LLMModelInfo{}, fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Model == "" {
		return model.LLMModelInfo{}, fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	if !IsSupportedModel(c.Model) {
		return model.LLMModelInfo{}, fmt.Errorf("unsupported zen model %q: %w", c.Model, pkgerrors.ErrNotValid)
	}

	info, _ := ModelByID(c.Model)

	return info, nil
}

// New creates a new OpenCode Zen provider.
func New(cfg Config) (llm.Provider, error) {
	info, err := cfg.defaults()
	if err != nil {
		return nil, fmt.Errorf("invalid zen provider config: %w", err)
	}

	return openaichat.New(openaichat.Config{
		TokenSource: cfg.TokenSource,
		BaseURL:     defaultBaseURL,
		Model:       cfg.Model,
		ModelInfo:   info,
		Tools:       cfg.Tools,
		ProviderID:  "zen",
		Client:      cfg.Client,
	})
}
