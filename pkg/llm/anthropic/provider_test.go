package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/anthropic"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

func TestNewAnthropic(t *testing.T) {
	tests := map[string]struct {
		cfg    anthropic.Config
		expErr bool
	}{
		"Valid config should succeed.": {
			cfg: anthropic.Config{TokenSource: anthropic.NewAPIKeyTokenSource("sk-ant-test"), Model: anthropic.ModelClaudeSonnet46},
		},
		"Missing token source should fail.": {
			cfg:    anthropic.Config{Model: anthropic.ModelClaudeSonnet46},
			expErr: true,
		},
		"Missing model should fail.": {
			cfg:    anthropic.Config{TokenSource: anthropic.NewAPIKeyTokenSource("sk-ant-test")},
			expErr: true,
		},
		"Unsupported model should fail.": {
			cfg:    anthropic.Config{TokenSource: anthropic.NewAPIKeyTokenSource("sk-ant-test"), Model: "claude-not-real"},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := anthropic.NewAnthropic(test.cfg)

			if test.expErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestAnthropicProviderCall(t *testing.T) {
	tests := map[string]struct {
		serverHandler http.HandlerFunc
		req           llm.Request
		expErr        bool
		expErrIs      error
		assert        func(t *testing.T, resp *llm.Response)
		assertRequest func(t *testing.T, req *http.Request, body []byte)
	}{
		"Text response should map correctly.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":       anthropic.ModelClaudeSonnet46,
				"stop_reason": "end_turn",
				"content": []map[string]any{{
					"type": "text",
					"text": "Hello from Claude",
				}},
				"usage": map[string]any{
					"input_tokens":                11,
					"output_tokens":               7,
					"cache_read_input_tokens":     2,
					"cache_creation_input_tokens": 3,
				},
			}),
			req: llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}}}},
			assert: func(t *testing.T, resp *llm.Response) {
				t.Helper()
				if len(resp.Message.Content) != 1 || resp.Message.Content[0].Text != "Hello from Claude" {
					t.Fatalf("unexpected content: %+v", resp.Message.Content)
				}
				if resp.Message.Metadata == nil {
					t.Fatal("expected metadata")
				}
				if resp.Message.Metadata.Provider != "anthropic" {
					t.Fatalf("expected provider anthropic, got %q", resp.Message.Metadata.Provider)
				}
				if resp.Message.Metadata.StopReason != model.StopReasonComplete {
					t.Fatalf("expected complete stop reason, got %q", resp.Message.Metadata.StopReason)
				}
				if resp.Message.Metadata.Usage == nil {
					t.Fatalf("unexpected usage: %+v", resp.Message.Metadata.Usage)
				}
				if resp.Message.Metadata.Usage.InputTokens != 11 {
					t.Fatalf("expected input tokens 11, got %d", resp.Message.Metadata.Usage.InputTokens)
				}
				if resp.Message.Metadata.Usage.CacheReadTokens != 2 || resp.Message.Metadata.Usage.CacheWriteTokens != 3 {
					t.Fatalf("unexpected cache usage: %+v", resp.Message.Metadata.Usage)
				}
				if resp.Message.Metadata.Usage.TotalTokens != 23 {
					t.Fatalf("expected total tokens 23, got %d", resp.Message.Metadata.Usage.TotalTokens)
				}
			},
			assertRequest: func(t *testing.T, req *http.Request, body []byte) {
				t.Helper()
				if req.Header.Get("x-api-key") != "sk-ant-test" {
					t.Fatalf("expected x-api-key header")
				}
				if req.Header.Get("anthropic-version") == "" {
					t.Fatalf("missing anthropic-version header")
				}

				var payload map[string]any
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Fatalf("unmarshal body: %v", err)
				}
				if payload["model"] != anthropic.ModelClaudeSonnet46 {
					t.Fatalf("unexpected model: %v", payload["model"])
				}
				if _, ok := payload["max_tokens"]; !ok {
					t.Fatalf("expected max_tokens in payload")
				}
			},
		},
		"Tool use response should map correctly.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":       anthropic.ModelClaudeSonnet46,
				"stop_reason": "tool_use",
				"content": []map[string]any{{
					"type":  "tool_use",
					"id":    "tool_1",
					"name":  "read",
					"input": map[string]any{"path": "main.go"},
				}},
			}),
			req: llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "read main.go"}}}}},
			assert: func(t *testing.T, resp *llm.Response) {
				t.Helper()
				if resp.Message.Metadata == nil || resp.Message.Metadata.StopReason != model.StopReasonToolUse {
					t.Fatalf("expected tool_use stop reason")
				}
				if len(resp.Message.ToolCallRequests) != 1 {
					t.Fatalf("expected one tool call, got %d", len(resp.Message.ToolCallRequests))
				}
				if resp.Message.ToolCallRequests[0].ToolID != "read" {
					t.Fatalf("expected tool ID read, got %s", resp.Message.ToolCallRequests[0].ToolID)
				}
			},
		},
		"Cache retention should add anthropic cache controls.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":       anthropic.ModelClaudeSonnet46,
				"stop_reason": "end_turn",
				"content": []map[string]any{{
					"type": "text",
					"text": "ok",
				}},
			}),
			req: llm.Request{
				SystemPrompt: "be concise",
				Messages: []model.Message{{
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}},
				}},
				Config: llm.RequestConfig{EnablePromptCache: true},
			},
			assertRequest: func(t *testing.T, _ *http.Request, body []byte) {
				t.Helper()

				var payload struct {
					System []struct {
						Type         string `json:"type"`
						Text         string `json:"text"`
						CacheControl struct {
							Type string `json:"type"`
							TTL  string `json:"ttl"`
						} `json:"cache_control"`
					} `json:"system"`
					Messages []struct {
						Role    string `json:"role"`
						Content []struct {
							Type         string `json:"type"`
							Text         string `json:"text"`
							CacheControl struct {
								Type string `json:"type"`
								TTL  string `json:"ttl"`
							} `json:"cache_control"`
						} `json:"content"`
					} `json:"messages"`
				}
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Fatalf("unmarshal body: %v", err)
				}

				if len(payload.System) != 1 {
					t.Fatalf("expected one system block, got %d", len(payload.System))
				}
				if payload.System[0].CacheControl.Type != "ephemeral" {
					t.Fatalf("expected system cache control ephemeral, got %+v", payload.System[0].CacheControl)
				}
				if payload.System[0].CacheControl.TTL != "" {
					t.Fatalf("expected system cache control ttl to be empty on non-anthropic base url, got %+v", payload.System[0].CacheControl)
				}

				if len(payload.Messages) != 1 || payload.Messages[0].Role != "user" {
					t.Fatalf("expected one user message, got %+v", payload.Messages)
				}
				if len(payload.Messages[0].Content) != 1 {
					t.Fatalf("expected one user content block, got %+v", payload.Messages[0].Content)
				}
				if payload.Messages[0].Content[0].CacheControl.Type != "ephemeral" {
					t.Fatalf("expected user cache control ephemeral, got %+v", payload.Messages[0].Content[0].CacheControl)
				}
			},
		},
		"API error should wrap llm error.": {
			serverHandler: jsonHandler(429, map[string]any{
				"type": "error",
				"error": map[string]any{
					"type":    "rate_limit_error",
					"message": "rate limit",
				},
			}),
			req:      llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}}}},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var capturedReq *http.Request
			var capturedBody []byte

			h := test.serverHandler
			if test.assertRequest != nil {
				original := h
				h = func(w http.ResponseWriter, r *http.Request) {
					capturedReq = r.Clone(r.Context())
					capturedBody, _ = io.ReadAll(r.Body)
					original(w, r)
				}
			}

			server := httptest.NewServer(h)
			defer server.Close()

			provider, err := anthropic.NewAnthropic(anthropic.Config{
				TokenSource: anthropic.NewAPIKeyTokenSource("sk-ant-test"),
				BaseURL:     server.URL,
				Model:       anthropic.ModelClaudeSonnet46,
				Client:      server.Client(),
			})
			if err != nil {
				t.Fatalf("new provider: %v", err)
			}

			resp, err := provider.Call(context.Background(), test.req)
			if test.expErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if test.expErrIs != nil && !errors.Is(err, test.expErrIs) {
					t.Fatalf("expected wrapped error %v, got %v", test.expErrIs, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if test.assertRequest != nil {
				test.assertRequest(t, capturedReq, capturedBody)
			}
			if test.assert != nil {
				test.assert(t, resp)
			}
		})
	}
}

func TestNewClaude(t *testing.T) {
	tests := map[string]struct {
		cfg    anthropic.ClaudeConfig
		expErr bool
	}{
		"Valid config should succeed.": {
			cfg: anthropic.ClaudeConfig{TokenSource: fakeTokenSource{token: "oauth"}, Model: anthropic.ModelClaudeSonnet46},
		},
		"Missing token source should fail.": {
			cfg:    anthropic.ClaudeConfig{Model: anthropic.ModelClaudeSonnet46},
			expErr: true,
		},
		"Missing model should fail.": {
			cfg:    anthropic.ClaudeConfig{TokenSource: fakeTokenSource{token: "oauth"}},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := anthropic.NewClaude(test.cfg)
			if test.expErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestClaudeProviderCall(t *testing.T) {
	readTool := staticTool{
		id:          "read",
		description: "Read a file",
		schema:      json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
	}

	var capturedReq *http.Request
	var capturedBody []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedReq = r.Clone(r.Context())
		capturedBody, _ = io.ReadAll(r.Body)

		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":       anthropic.ModelClaudeSonnet46,
			"stop_reason": "tool_use",
			"content": []map[string]any{{
				"type":  "tool_use",
				"id":    "tool_1",
				"name":  "Read",
				"input": map[string]any{"path": "main.go"},
			}},
		})
	}))
	defer server.Close()

	provider, err := anthropic.NewClaude(anthropic.ClaudeConfig{
		TokenSource: fakeTokenSource{token: "oauth-token"},
		BaseURL:     server.URL,
		Model:       anthropic.ModelClaudeSonnet46,
		Tools:       []tool.Tool{readTool},
		Client:      server.Client(),
	})
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}

	resp, err := provider.Call(context.Background(), llm.Request{
		SystemPrompt: "Keep it short",
		Messages: []model.Message{{
			Kind:    model.MessageKindUser,
			Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "read main.go"}},
		}},
	})
	if err != nil {
		t.Fatalf("call: %v", err)
	}

	if capturedReq.Header.Get("Authorization") != "Bearer oauth-token" {
		t.Fatalf("expected bearer auth header")
	}
	if capturedReq.Header.Get("anthropic-beta") == "" {
		t.Fatalf("expected anthropic-beta header")
	}
	if capturedReq.Header.Get("x-app") != "cli" {
		t.Fatalf("expected x-app=cli")
	}

	var payload struct {
		System string `json:"system"`
		Tools  []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(capturedBody, &payload); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	if payload.System == "" || !strings.Contains(payload.System, "Claude Code") {
		t.Fatalf("expected Claude identity in system prompt")
	}
	if len(payload.Tools) != 1 || payload.Tools[0].Name != "Read" {
		t.Fatalf("expected normalized tool name Read, got %+v", payload.Tools)
	}

	if len(resp.Message.ToolCallRequests) != 1 || resp.Message.ToolCallRequests[0].ToolID != "read" {
		t.Fatalf("expected restored tool ID read, got %+v", resp.Message.ToolCallRequests)
	}
	if resp.Message.Metadata == nil || resp.Message.Metadata.Provider != "anthropic-claude" {
		t.Fatalf("unexpected provider metadata: %+v", resp.Message.Metadata)
	}
}

func TestAnthropicProviderModelInfo(t *testing.T) {
	provider, err := anthropic.NewAnthropic(anthropic.Config{
		TokenSource: anthropic.NewAPIKeyTokenSource("sk-ant-test"),
		Model:       anthropic.ModelClaudeSonnet46,
	})
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}

	info := provider.ModelInfo()
	if info.ID != anthropic.ModelClaudeSonnet46 {
		t.Fatalf("expected model id %q, got %q", anthropic.ModelClaudeSonnet46, info.ID)
	}
	if info.ContextWindow <= 0 {
		t.Fatalf("expected positive context window, got %d", info.ContextWindow)
	}
}

func TestClaudeProviderModelInfo(t *testing.T) {
	provider, err := anthropic.NewClaude(anthropic.ClaudeConfig{
		TokenSource: fakeTokenSource{token: "oauth"},
		Model:       anthropic.ModelClaudeSonnet46,
	})
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}

	info := provider.ModelInfo()
	if info.ID != anthropic.ModelClaudeSonnet46 {
		t.Fatalf("expected model id %q, got %q", anthropic.ModelClaudeSonnet46, info.ID)
	}
	if info.ContextWindow <= 0 {
		t.Fatalf("expected positive context window, got %d", info.ContextWindow)
	}
}

type fakeTokenSource struct {
	token string
}

func (f fakeTokenSource) Token(_ context.Context) (string, error) {
	return f.token, nil
}

func jsonHandler(status int, body any) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(body)
	}
}

type staticTool struct {
	id          string
	description string
	schema      json.RawMessage
}

func (t staticTool) ID() string              { return t.id }
func (t staticTool) Description() string     { return t.description }
func (t staticTool) Schema() json.RawMessage { return t.schema }
func (t staticTool) Execute(context.Context, json.RawMessage) (*tool.Result, error) {
	return nil, nil
}
