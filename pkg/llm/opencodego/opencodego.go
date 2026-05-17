package opencodego

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/internal/openaichat"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

//go:generate go run ../internal/cmd/genmodels -target opencode-go -out ./models_gen.go

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

func (c *Config) defaults() error {
	if c.TokenSource == nil {
		return fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.BaseURL) == "" {
		c.BaseURL = defaultBaseURL
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	return nil
}

// New creates a new OpenCode Go provider.
func New(cfg Config) (llm.Provider, error) {
	err := cfg.defaults()
	if err != nil {
		return nil, fmt.Errorf("invalid opencode-go provider config: %w", err)
	}

	info, ok := ModelByID(cfg.Model)
	if !ok {
		return nil, fmt.Errorf("unsupported opencode-go model %q: %w", cfg.Model, pkgerrors.ErrNotValid)
	}

	return openaichat.New(openaichat.Config{
		TokenSource: cfg.TokenSource,
		BaseURL:     cfg.BaseURL,
		Model:       cfg.Model,
		ModelInfo:   info,
		ProviderID:  providerID,
		Client:      cfg.Client,
	})
}
