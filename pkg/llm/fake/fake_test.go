package fake_test

import (
	"context"
	"fmt"
	"testing"

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
			p := fake.NewProvider(test.fn)

			got, err := p.Call(context.Background(), test.req)

			if test.expErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if got.Message.Kind != test.expResp.Message.Kind {
				t.Errorf("expected kind %q, got %q", test.expResp.Message.Kind, got.Message.Kind)
			}

			if len(got.Message.Content) != len(test.expResp.Message.Content) {
				t.Errorf("expected %d content parts, got %d", len(test.expResp.Message.Content), len(got.Message.Content))
				return
			}

			for i, part := range got.Message.Content {
				if part.Text != test.expResp.Message.Content[i].Text {
					t.Errorf("content[%d]: expected text %q, got %q", i, test.expResp.Message.Content[i].Text, part.Text)
				}
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
			p := fake.NewEchoProvider()

			got, err := p.Call(context.Background(), test.req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got.Message.Kind != model.MessageKindLLM {
				t.Errorf("expected kind %q, got %q", model.MessageKindLLM, got.Message.Kind)
			}

			if got.Message.Metadata == nil || got.Message.Metadata.StopReason != model.StopReasonComplete {
				t.Error("expected StopReasonComplete in metadata")
			}

			if test.expHasContent {
				if len(got.Message.Content) == 0 {
					t.Fatal("expected content, got none")
				}
				if got.Message.Content[0].Text != test.expContentText {
					t.Errorf("expected text %q, got %q", test.expContentText, got.Message.Content[0].Text)
				}
			} else {
				if len(got.Message.Content) != 0 {
					t.Errorf("expected no content, got %d parts", len(got.Message.Content))
				}
			}
		})
	}
}

func TestProviderModelInfo(t *testing.T) {
	exp := model.LLMModelInfo{ID: "fake-model", ContextWindow: 1234, MaxOutputTokens: 99}

	p := fake.NewProviderWithModelInfo(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
		return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM}}, nil
	}, exp)

	got := p.ModelInfo()
	if got.ID != exp.ID || got.ContextWindow != exp.ContextWindow || got.MaxOutputTokens != exp.MaxOutputTokens {
		t.Fatalf("expected model info %+v, got %+v", exp, got)
	}
}
