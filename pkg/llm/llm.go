package llm

import (
	"context"
	"encoding/json"

	"github.com/slok/gosimov/pkg/model"
)

// ToolDefinition is the tool metadata sent to the LLM provider on each call.
//
// This is a provider-facing DTO; actual tool execution remains in the agent
// loop via [tool.Tool].
type ToolDefinition struct {
	ID          string
	Description string
	Schema      json.RawMessage
}

// Provider sends messages to an LLM and returns responses.
type Provider interface {
	// Call sends a request to the LLM and returns the full response (blocking).
	Call(ctx context.Context, req Request) (*Response, error)
	// ModelInfo returns metadata for the model configured on this provider.
	ModelInfo() model.LLMModelInfo
}

// Request is what gets sent to the LLM.
type Request struct {
	SystemPrompt string
	// SessionID identifies the conversation/session making this request.
	// Providers can use this for stable per-session behaviors (e.g. prompt cache keys).
	SessionID string
	Messages  []model.Message
	// Tools are the tool definitions available for this specific LLM call.
	//
	// This enables per-turn/per-iteration dynamic tool descriptions and schemas.
	Tools  []ToolDefinition
	Config RequestConfig
}

// RequestConfig holds LLM call configuration.
// Fields will be added as providers need them (temperature, max tokens, etc.).
type RequestConfig struct {
	// MaxTokens limits the maximum number of output tokens.
	// 0 means provider default.
	MaxTokens int
	// EnablePromptCache enables provider-side prompt caching when supported.
	// Providers map this hint to their native cache mechanism.
	EnablePromptCache bool
}

// Response is what comes back from the LLM.
//
// Message has Kind, Content, ToolCallRequests, and Metadata set by the provider.
// The caller is responsible for setting ID and CreatedAt before storing.
type Response struct {
	Message model.Message
}

// NoopProvider is a Provider that returns an empty response with a "noop" model.
var NoopProvider = noopProvider(false)

type noopProvider bool

func (p *noopProvider) Call(ctx context.Context, req Request) (*Response, error) {
	return &Response{
		Message: model.Message{
			Kind: model.MessageKindLLM,
			Metadata: &model.MessageMetadata{
				StopReason: model.StopReasonComplete,
				Model:      "noop",
				Provider:   "noop",
			},
		},
	}, nil
}

func (p *noopProvider) ModelInfo() model.LLMModelInfo {
	return model.LLMModelInfo{Name: "noop"}
}
