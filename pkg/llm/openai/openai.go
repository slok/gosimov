package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/internal/openaichat"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

const (
	defaultOpenAIBaseURL = "https://api.openai.com/v1"
	openAIProviderID     = "openai"
)

type chatErrorResponse struct {
	Error chatErrorDetail `json:"error"`
}

type chatErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// OpenAIConfig configures the OpenAI chat completions provider.
type OpenAIConfig struct {
	// TokenSource provides bearer tokens for authentication (required).
	// Use [NewAPIKeyTokenSource] for static API key auth.
	TokenSource TokenSource
	// BaseURL overrides OpenAI API base URL.
	// Defaults to "https://api.openai.com/v1".
	BaseURL string
	// Model is the model ID to use (required).
	Model string
	// Tools are the tools available for the LLM to call (optional).
	Tools []tool.Tool
	// Client is the HTTP client used for API calls (optional).
	// Defaults to [http.DefaultClient].
	Client *http.Client
}

func (c *OpenAIConfig) defaults() (model.LLMModelInfo, error) {
	if c.TokenSource == nil {
		return model.LLMModelInfo{}, fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Model == "" {
		return model.LLMModelInfo{}, fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if c.BaseURL == "" {
		c.BaseURL = defaultOpenAIBaseURL
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	if !IsSupportedOpenAIModel(c.Model) {
		return model.LLMModelInfo{}, fmt.Errorf("unsupported openai model %q: %w", c.Model, pkgerrors.ErrNotValid)
	}

	info, _ := OpenAIModelInfo(c.Model)

	return info, nil
}

// NewOpenAI creates a new OpenAI chat completions provider.
func NewOpenAI(cfg OpenAIConfig) (llm.Provider, error) {
	info, err := cfg.defaults()
	if err != nil {
		return nil, fmt.Errorf("invalid openai provider config: %w", err)
	}

	return openaichat.New(openaichat.Config{
		TokenSource: cfg.TokenSource,
		BaseURL:     cfg.BaseURL,
		Model:       cfg.Model,
		ModelInfo:   info,
		Tools:       cfg.Tools,
		ProviderID:  openAIProviderID,
		Client:      cfg.Client,
	})
}

// parseAPIError extracts an error from an API error response.
func parseAPIError(statusCode int, body []byte) error {
	var errResp chatErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil {
		return fmt.Errorf("api error (status %d): %s: %w", statusCode, string(body), pkgerrors.ErrLLMError)
	}

	msg := strings.TrimSpace(errResp.Error.Message)
	errType := strings.TrimSpace(errResp.Error.Type)
	if msg == "" {
		msg = strings.TrimSpace(string(body))
	}
	if msg == "" {
		msg = "unknown api error"
	}

	return fmt.Errorf("api error (status %d, type=%s): %s: %w",
		statusCode, errType, msg, pkgerrors.ErrLLMError)
}
