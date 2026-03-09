package fake_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/fake"
	"github.com/slok/gosimov/pkg/model"
)

func TestProviderCall(t *testing.T) {
	tests := map[string]struct {
		fn      llm.CompleteFn
		req     llm.Request
		expResp *llm.Response
		expErr  bool
	}{
		"Custom function should be called and return its response.": {
			fn: func(_ context.Context, _ llm.Request) (*llm.Response, error) {
				return &llm.Response{
					Message: model.Message{
						Kind:    model.MessageKindLLM,
						Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "custom response"}},
					},
				}, nil
			},
			req: llm.Request{},
			expResp: &llm.Response{
				Message: model.Message{
					Kind:    model.MessageKindLLM,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "custom response"}},
				},
			},
		},

		"Custom function error should propagate.": {
			fn: func(_ context.Context, _ llm.Request) (*llm.Response, error) {
				return nil, fmt.Errorf("something broke")
			},
			req:    llm.Request{},
			expErr: true,
		},

		"Custom function should receive the request.": {
			fn: func(_ context.Context, req llm.Request) (*llm.Response, error) {
				return &llm.Response{
					Message: model.Message{
						Kind:    model.MessageKindLLM,
						Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: req.SystemPrompt}},
					},
				}, nil
			},
			req: llm.Request{SystemPrompt: "you are helpful"},
			expResp: &llm.Response{
				Message: model.Message{
					Kind:    model.MessageKindLLM,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "you are helpful"}},
				},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			p := fake.NewProvider(test.fn)
			got, err := p.Call(context.Background(), test.req)

			if test.expErr {
				require.Error(err)
				return
			}

			require.NoError(err)
			assert.Equal(test.expResp.Message.Kind, got.Message.Kind)
			require.Len(got.Message.Content, len(test.expResp.Message.Content))

			for i, part := range got.Message.Content {
				assert.Equal(test.expResp.Message.Content[i].Text, part.Text)
			}
		})
	}
}

func TestEchoProviderCall(t *testing.T) {
	tests := map[string]struct {
		req            llm.Request
		expContentText string
		expHasContent  bool
	}{
		"Should echo the last user message text.": {
			req: llm.Request{
				Messages: []model.Message{
					{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
				},
			},
			expContentText: "hello",
			expHasContent:  true,
		},

		"Should echo the last user message when multiple exist.": {
			req: llm.Request{
				Messages: []model.Message{
					{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "first"}}},
					{Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "response"}}},
					{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "second"}}},
				},
			},
			expContentText: "second",
			expHasContent:  true,
		},

		"Should return empty content when no user messages exist.": {
			req: llm.Request{
				Messages: []model.Message{
					{Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "orphan"}}},
				},
			},
			expHasContent: false,
		},

		"Should return empty content when messages are empty.": {
			req:           llm.Request{},
			expHasContent: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			p := fake.NewEchoProvider()
			got, err := p.Call(context.Background(), test.req)
			require.NoError(err)

			assert.Equal(model.MessageKindLLM, got.Message.Kind)
			require.NotNil(got.Message.Metadata)
			assert.Equal(model.StopReasonComplete, got.Message.Metadata.StopReason)

			if test.expHasContent {
				require.NotEmpty(got.Message.Content)
				assert.Equal(test.expContentText, got.Message.Content[0].Text)
			} else {
				assert.Empty(got.Message.Content)
			}
		})
	}
}

func TestProviderModelInfo(t *testing.T) {
	tests := map[string]struct {
		modelInfo    model.LLMModelInfo
		expModelInfo model.LLMModelInfo
	}{
		"Should return the configured model info.": {
			modelInfo:    model.LLMModelInfo{ID: "fake-model", ContextWindow: 1234, MaxOutputTokens: 99},
			expModelInfo: model.LLMModelInfo{ID: "fake-model", ContextWindow: 1234, MaxOutputTokens: 99},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			p := fake.NewProviderWithModelInfo(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
				return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM}}, nil
			}, test.modelInfo)

			got := p.ModelInfo()
			assert.Equal(test.expModelInfo, got)
		})
	}
}
