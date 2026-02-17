package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	gllm "github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/openai"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

func TestNewChatGPT(t *testing.T) {
	tests := map[string]struct {
		cfg    openai.ChatGPTConfig
		expErr bool
	}{
		"Valid token source config should succeed.": {
			cfg: openai.ChatGPTConfig{
				TokenSource: fakeTokenSource{token: jwtWithAccount("acct-123")},
				Model:       "gpt-5.3-codex",
			},
		},
		"Missing token source should fail.": {
			cfg: openai.ChatGPTConfig{
				Model: "gpt-5.3-codex",
			},
			expErr: true,
		},
		"Missing model should fail.": {
			cfg: openai.ChatGPTConfig{
				TokenSource: fakeTokenSource{token: jwtWithAccount("acct-123")},
			},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := openai.NewChatGPT(test.cfg)
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

func TestCodexResponsesCall(t *testing.T) {
	tests := map[string]struct {
		token         string
		serverHandler http.HandlerFunc
		req           gllm.Request
		expErr        bool
		expErrIs      error
		assert        func(t *testing.T, resp *gllm.Response)
		assertRequest func(t *testing.T, req *http.Request, body []byte)
	}{
		"Should map text response and set codex headers.": {
			token: jwtWithAccount("acct-123"),
			serverHandler: jsonHandler(200, map[string]any{
				"model":  "gpt-5.3-codex",
				"status": "completed",
				"output": []map[string]any{{
					"type": "message",
					"role": "assistant",
					"content": []map[string]any{{
						"type": "output_text",
						"text": "Hello from codex",
					}},
				}},
				"usage": map[string]any{
					"input_tokens":  10,
					"output_tokens": 5,
					"input_tokens_details": map[string]any{
						"cached_tokens": 2,
					},
				},
			}),
			req: gllm.Request{
				SystemPrompt: "be concise",
				Messages: []model.Message{{
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}},
				}},
			},
			assertRequest: func(t *testing.T, req *http.Request, body []byte) {
				t.Helper()
				if req.URL.Path != "/codex/responses" {
					t.Fatalf("expected /codex/responses path, got %s", req.URL.Path)
				}
				if req.Header.Get("Authorization") != "Bearer "+jwtWithAccount("acct-123") {
					t.Fatalf("missing auth header")
				}
				if req.Header.Get("chatgpt-account-id") != "acct-123" {
					t.Fatalf("missing chatgpt account id header")
				}
				if req.Header.Get("OpenAI-Beta") != "responses=experimental" {
					t.Fatalf("missing OpenAI-Beta header")
				}
				if req.Header.Get("originator") != "pi" {
					t.Fatalf("expected originator pi")
				}

				var payload map[string]any
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Fatalf("unmarshal body: %v", err)
				}
				if payload["model"] != "gpt-5.3-codex" {
					t.Fatalf("expected model gpt-5.3-codex, got %v", payload["model"])
				}
			},
			assert: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				if len(resp.Message.Content) != 1 || resp.Message.Content[0].Text != "Hello from codex" {
					t.Fatalf("unexpected content: %+v", resp.Message.Content)
				}
				if resp.Message.Metadata == nil {
					t.Fatal("expected metadata")
				}
				if resp.Message.Metadata.Provider != "openai-codex" {
					t.Fatalf("expected provider openai-codex, got %s", resp.Message.Metadata.Provider)
				}
				if resp.Message.Metadata.StopReason != model.StopReasonComplete {
					t.Fatalf("unexpected stop reason: %s", resp.Message.Metadata.StopReason)
				}
				if resp.Message.Metadata.Usage == nil {
					t.Fatalf("unexpected usage: %+v", resp.Message.Metadata.Usage)
				}
				if resp.Message.Metadata.Usage.InputTokens != 8 {
					t.Fatalf("expected non-cached input tokens 8, got %d", resp.Message.Metadata.Usage.InputTokens)
				}
				if resp.Message.Metadata.Usage.CacheReadTokens != 2 {
					t.Fatalf("expected cache read tokens 2, got %d", resp.Message.Metadata.Usage.CacheReadTokens)
				}
				if resp.Message.Metadata.Usage.TotalTokens != 15 {
					t.Fatalf("expected total tokens 15, got %d", resp.Message.Metadata.Usage.TotalTokens)
				}
			},
		},
		"Should map function call output as tool use.": {
			token: jwtWithAccount("acct-123"),
			serverHandler: jsonHandler(200, map[string]any{
				"model":  "gpt-5.3-codex",
				"status": "completed",
				"output": []map[string]any{{
					"type":      "function_call",
					"call_id":   "call_1",
					"name":      "read",
					"arguments": `{"path":"main.go"}`,
				}},
			}),
			req: gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "read main.go"}}}}},
			assert: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				if resp.Message.Metadata == nil || resp.Message.Metadata.StopReason != model.StopReasonToolUse {
					t.Fatalf("expected tool_use stop reason")
				}
				if len(resp.Message.ToolCallRequests) != 1 || resp.Message.ToolCallRequests[0].ToolID != "read" {
					t.Fatalf("unexpected tool calls: %+v", resp.Message.ToolCallRequests)
				}
			},
		},
		"Should include prompt cache key in request.": {
			token: jwtWithAccount("acct-123"),
			serverHandler: jsonHandler(200, map[string]any{
				"model":  "gpt-5.3-codex",
				"status": "completed",
				"output": []map[string]any{{
					"type": "message",
					"role": "assistant",
					"content": []map[string]any{{
						"type": "output_text",
						"text": "cached",
					}},
				}},
			}),
			req: gllm.Request{
				SessionID: "s_123",
				Messages:  []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}}},
				Config:    gllm.RequestConfig{EnablePromptCache: true},
			},
			assertRequest: func(t *testing.T, _ *http.Request, body []byte) {
				t.Helper()

				var payload struct {
					PromptCacheKey       string `json:"prompt_cache_key"`
					PromptCacheRetention string `json:"prompt_cache_retention"`
				}
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Fatalf("unmarshal body: %v", err)
				}

				if payload.PromptCacheKey != "gosimov-sess-s_123" {
					t.Fatalf("expected prompt_cache_key gosimov-sess-s_123, got %q", payload.PromptCacheKey)
				}
				if payload.PromptCacheRetention != "" {
					t.Fatalf("expected empty prompt_cache_retention on chatgpt backend, got %q", payload.PromptCacheRetention)
				}
			},
		},
		"Should parse SSE completed event response.": {
			token: jwtWithAccount("acct-123"),
			serverHandler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\"}}\n\n"))
				_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.3-codex\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"SSE works\"}]}]}}\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
			},
			req: gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}}}},
			assert: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				if len(resp.Message.Content) != 1 || resp.Message.Content[0].Text != "SSE works" {
					t.Fatalf("unexpected content: %+v", resp.Message.Content)
				}
			},
		},
		"Missing account ID in token should fail.": {
			token:         "abc.def.ghi",
			serverHandler: jsonHandler(200, map[string]any{"model": "gpt-5.3-codex", "status": "completed", "output": []any{}}),
			req:           gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}}}},
			expErr:        true,
			expErrIs:      pkgerrors.ErrNotValid,
		},
		"API error should wrap llm error.": {
			token: jwtWithAccount("acct-123"),
			serverHandler: jsonHandler(429, map[string]any{
				"error": map[string]any{"message": "rate limit", "type": "rate_limit"},
			}),
			req:      gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}}}},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var capturedReq *http.Request
			var capturedBody []byte

			handler := test.serverHandler
			if test.assertRequest != nil {
				original := handler
				handler = func(w http.ResponseWriter, r *http.Request) {
					capturedReq = r.Clone(r.Context())
					capturedBody, _ = io.ReadAll(r.Body)
					original(w, r)
				}
			}

			server := httptest.NewServer(handler)
			defer server.Close()

			provider, err := openai.NewChatGPT(openai.ChatGPTConfig{
				TokenSource: fakeTokenSource{token: test.token},
				BaseURL:     server.URL,
				Model:       "gpt-5.3-codex",
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

func TestChatGPTProviderModelInfo(t *testing.T) {
	provider, err := openai.NewChatGPT(openai.ChatGPTConfig{
		TokenSource: fakeTokenSource{token: jwtWithAccount("acct-123")},
		Model:       "gpt-5.3-codex",
	})
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}

	info := provider.ModelInfo()
	if info.ID != "gpt-5.3-codex" {
		t.Fatalf("expected model id gpt-5.3-codex, got %q", info.ID)
	}
	if info.ContextWindow <= 0 {
		t.Fatalf("expected positive context window, got %d", info.ContextWindow)
	}
	if info.MaxOutputTokens <= 0 {
		t.Fatalf("expected positive max output tokens, got %d", info.MaxOutputTokens)
	}
}

func jwtWithAccount(accountID string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"` + accountID + `"}}`))
	return strings.Join([]string{header, payload, "sig"}, ".")
}
