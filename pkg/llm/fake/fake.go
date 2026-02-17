package fake

import (
	"context"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/model"
)

// Provider is a configurable LLM provider that delegates to a function.
// Use [NewProvider] with a custom function, or [NewEchoProvider] for a
// pre-configured echo implementation.
type Provider struct {
	fn        llm.CompleteFn
	modelInfo model.LLMModelInfo
}

// NewProvider creates a Provider that delegates to the given function.
func NewProvider(fn llm.CompleteFn) *Provider {
	return &Provider{fn: fn}
}

// NewProviderWithModelInfo creates a Provider with explicit model metadata.
func NewProviderWithModelInfo(fn llm.CompleteFn, modelInfo model.LLMModelInfo) *Provider {
	return &Provider{fn: fn, modelInfo: modelInfo}
}

// NewEchoProvider creates a Provider that echoes the last user message
// content back as an LLM response.
func NewEchoProvider() *Provider {
	return NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
		var parts []model.ContentPart

		// Find the last user message and echo its content.
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Kind == model.MessageKindUser {
				parts = req.Messages[i].Content
				break
			}
		}

		return &llm.Response{
			Message: model.Message{
				Kind:    model.MessageKindLLM,
				Content: parts,
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonComplete,
				},
			},
		}, nil
	})
}

// Call implements [llm.Provider].
func (p *Provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	return p.fn(ctx, req)
}

// ModelInfo implements [llm.Provider].
func (p *Provider) ModelInfo() model.LLMModelInfo {
	return p.modelInfo
}
