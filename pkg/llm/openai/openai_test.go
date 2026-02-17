package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	gllm "github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/openai"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

func TestNewOpenAI(t *testing.T) {
	tests := map[string]struct {
		cfg    openai.OpenAIConfig
		expErr bool
	}{
		"Valid config should succeed.": {
			cfg: openai.OpenAIConfig{
				TokenSource: openai.NewAPIKeyTokenSource("sk-test"),
				Model:       "gpt-4o",
			},
		},

		"Valid token source config should succeed.": {
			cfg: openai.OpenAIConfig{
				TokenSource: fakeTokenSource{token: "oauth-token"},
				Model:       "gpt-4o",
			},
		},

		"Missing token source should fail.": {
			cfg: openai.OpenAIConfig{
				Model: "gpt-4o",
			},
			expErr: true,
		},

		"Missing model should fail.": {
			cfg: openai.OpenAIConfig{
				TokenSource: openai.NewAPIKeyTokenSource("sk-test"),
			},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := openai.NewOpenAI(test.cfg)

			if test.expErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestProviderCall(t *testing.T) {
	tests := map[string]struct {
		// serverHandler is the HTTP handler for the mock server.
		serverHandler http.HandlerFunc
		// req is the LLM request to send.
		req func() gllm.Request
		// expErr is true if Call should return an error.
		expErr bool
		// expErrIs is the sentinel error to check with errors.Is.
		expErrIs error
		// assert runs custom assertions on the response.
		assert func(t *testing.T, resp *gllm.Response)
		// assertRequest runs assertions on the HTTP request received by the server.
		assertRequest func(t *testing.T, httpReq *http.Request, body []byte)
	}{
		"Simple text response should be converted correctly.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model": "glm-5-free",
				"choices": []map[string]any{{
					"message":       map[string]any{"role": "assistant", "content": "Hello! How can I help?"},
					"finish_reason": "stop",
				}},
				"usage": map[string]any{
					"prompt_tokens":     15,
					"completion_tokens": 8,
				},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					SystemPrompt: "be helpful",
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			assert: func(t *testing.T, resp *gllm.Response) {
				t.Helper()

				if resp.Message.Kind != model.MessageKindLLM {
					t.Errorf("expected kind %q, got %q", model.MessageKindLLM, resp.Message.Kind)
				}

				if len(resp.Message.Content) != 1 {
					t.Fatalf("expected 1 content part, got %d", len(resp.Message.Content))
				}

				if resp.Message.Content[0].Text != "Hello! How can I help?" {
					t.Errorf("expected text %q, got %q", "Hello! How can I help?", resp.Message.Content[0].Text)
				}

				if resp.Message.Metadata == nil {
					t.Fatal("expected metadata")
				}

				if resp.Message.Metadata.StopReason != model.StopReasonComplete {
					t.Errorf("expected stop reason %q, got %q", model.StopReasonComplete, resp.Message.Metadata.StopReason)
				}

				if resp.Message.Metadata.Model != "glm-5-free" {
					t.Errorf("expected model %q, got %q", "glm-5-free", resp.Message.Metadata.Model)
				}

				if resp.Message.Metadata.Provider != "openai" {
					t.Errorf("expected provider %q, got %q", "openai", resp.Message.Metadata.Provider)
				}

				if resp.Message.Metadata.Usage == nil {
					t.Fatal("expected usage")
				}

				if resp.Message.Metadata.Usage.InputTokens != 15 {
					t.Errorf("expected input tokens 15, got %d", resp.Message.Metadata.Usage.InputTokens)
				}

				if resp.Message.Metadata.Usage.OutputTokens != 8 {
					t.Errorf("expected output tokens 8, got %d", resp.Message.Metadata.Usage.OutputTokens)
				}

				if resp.Message.Metadata.Usage.TotalTokens != 23 {
					t.Errorf("expected total tokens 23, got %d", resp.Message.Metadata.Usage.TotalTokens)
				}
			},
		},

		"Cached prompt tokens should be mapped as non-cached input plus cache read.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model": "gpt-4o",
				"choices": []map[string]any{{
					"message":       map[string]any{"role": "assistant", "content": "ok"},
					"finish_reason": "stop",
				}},
				"usage": map[string]any{
					"prompt_tokens":     20,
					"completion_tokens": 9,
					"prompt_tokens_details": map[string]any{
						"cached_tokens": 6,
					},
				},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			assert: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				require.NotNil(t, resp.Message.Metadata)
				require.NotNil(t, resp.Message.Metadata.Usage)
				assert.Equal(t, 14, resp.Message.Metadata.Usage.InputTokens)
				assert.Equal(t, 9, resp.Message.Metadata.Usage.OutputTokens)
				assert.Equal(t, 6, resp.Message.Metadata.Usage.CacheReadTokens)
				assert.Equal(t, 29, resp.Message.Metadata.Usage.TotalTokens)
			},
		},

		"Tool call response should be converted correctly.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model": "gpt-4o",
				"choices": []map[string]any{{
					"message": map[string]any{
						"role": "assistant",
						"tool_calls": []map[string]any{{
							"id":   "call_abc123",
							"type": "function",
							"function": map[string]any{
								"name":      "read",
								"arguments": `{"path":"main.go"}`,
							},
						}},
					},
					"finish_reason": "tool_calls",
				}},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "read main.go"}}},
					},
				}
			},
			assert: func(t *testing.T, resp *gllm.Response) {
				t.Helper()

				if resp.Message.Metadata.StopReason != model.StopReasonToolUse {
					t.Errorf("expected stop reason %q, got %q", model.StopReasonToolUse, resp.Message.Metadata.StopReason)
				}

				if len(resp.Message.ToolCallRequests) != 1 {
					t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCallRequests))
				}

				tc := resp.Message.ToolCallRequests[0]
				if tc.ID != "call_abc123" {
					t.Errorf("expected tool call id %q, got %q", "call_abc123", tc.ID)
				}
				if tc.ToolID != "read" {
					t.Errorf("expected tool id %q, got %q", "read", tc.ToolID)
				}

				var args map[string]string
				if err := json.Unmarshal(tc.Arguments, &args); err != nil {
					t.Fatalf("expected valid JSON arguments: %v", err)
				}
				if args["path"] != "main.go" {
					t.Errorf("expected path %q, got %q", "main.go", args["path"])
				}
			},
		},

		"Request should include authorization header and correct model.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":   "test",
				"choices": []map[string]any{},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					SystemPrompt: "be helpful",
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			assertRequest: func(t *testing.T, httpReq *http.Request, body []byte) {
				t.Helper()

				if httpReq.Header.Get("Authorization") != "Bearer sk-test-key" {
					t.Errorf("expected authorization header, got %q", httpReq.Header.Get("Authorization"))
				}

				if httpReq.Header.Get("Content-Type") != "application/json" {
					t.Errorf("expected content-type application/json, got %q", httpReq.Header.Get("Content-Type"))
				}

				var reqBody map[string]json.RawMessage
				if err := json.Unmarshal(body, &reqBody); err != nil {
					t.Fatalf("failed to unmarshal body: %v", err)
				}

				var reqModel string
				if err := json.Unmarshal(reqBody["model"], &reqModel); err != nil {
					t.Fatalf("failed to unmarshal model: %v", err)
				}
				if reqModel != "gpt-4o" {
					t.Errorf("expected model %q, got %q", "gpt-4o", reqModel)
				}
			},
		},

		"Request should use OAuth token source authorization header.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":   "test",
				"choices": []map[string]any{},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			assertRequest: func(t *testing.T, httpReq *http.Request, _ []byte) {
				t.Helper()
				if httpReq.Header.Get("Authorization") != "Bearer oauth-token" {
					t.Errorf("expected authorization header with oauth token, got %q", httpReq.Header.Get("Authorization"))
				}
			},
		},

		"Request should include max_tokens when set in request config.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":   "test",
				"choices": []map[string]any{},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
					Config: gllm.RequestConfig{MaxTokens: 123},
				}
			},
			assertRequest: func(t *testing.T, _ *http.Request, body []byte) {
				t.Helper()

				var reqBody map[string]json.RawMessage
				if err := json.Unmarshal(body, &reqBody); err != nil {
					t.Fatalf("failed to unmarshal body: %v", err)
				}

				var maxTokens int
				if err := json.Unmarshal(reqBody["max_tokens"], &maxTokens); err != nil {
					t.Fatalf("failed to unmarshal max_tokens: %v", err)
				}

				if maxTokens != 123 {
					t.Errorf("expected max_tokens %d, got %d", 123, maxTokens)
				}
			},
		},

		"Request should include stable prompt_cache_key when cache enabled.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":   "test",
				"choices": []map[string]any{},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					SessionID: "s_abc123",
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
					Config: gllm.RequestConfig{EnablePromptCache: true},
				}
			},
			assertRequest: func(t *testing.T, _ *http.Request, body []byte) {
				t.Helper()

				var reqBody map[string]json.RawMessage
				if err := json.Unmarshal(body, &reqBody); err != nil {
					t.Fatalf("failed to unmarshal body: %v", err)
				}

				var promptCacheKey string
				if err := json.Unmarshal(reqBody["prompt_cache_key"], &promptCacheKey); err != nil {
					t.Fatalf("failed to unmarshal prompt_cache_key: %v", err)
				}

				if promptCacheKey != "gosimov-sess-s_abc123" {
					t.Errorf("expected prompt_cache_key %q, got %q", "gosimov-sess-s_abc123", promptCacheKey)
				}
			},
		},

		"API error 401 should return LLM error.": {
			serverHandler: jsonHandler(401, map[string]any{
				"error": map[string]any{
					"message": "invalid api key",
					"type":    "invalid_request_error",
					"code":    "invalid_api_key",
				},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},

		"API error 429 should return LLM error.": {
			serverHandler: jsonHandler(429, map[string]any{
				"error": map[string]any{
					"message": "rate limit exceeded",
					"type":    "rate_limit_error",
				},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},

		"API error with non-JSON body should still return LLM error.": {
			serverHandler: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(500)
				_, _ = w.Write([]byte("internal server error"))
			},
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},

		"Empty choices should return valid response with no content.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":   "test",
				"choices": []map[string]any{},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			assert: func(t *testing.T, resp *gllm.Response) {
				t.Helper()

				if resp.Message.Kind != model.MessageKindLLM {
					t.Errorf("expected kind %q, got %q", model.MessageKindLLM, resp.Message.Kind)
				}

				if len(resp.Message.Content) != 0 {
					t.Errorf("expected no content, got %d parts", len(resp.Message.Content))
				}
			},
		},

		"Context cancellation should return error.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model":   "test",
				"choices": []map[string]any{},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			// This test is handled specially below.
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var (
				capturedReq  *http.Request
				capturedBody []byte
			)

			handler := test.serverHandler
			if test.assertRequest != nil {
				// Wrap handler to capture the request for assertion.
				original := handler
				handler = func(w http.ResponseWriter, r *http.Request) {
					capturedReq = r.Clone(r.Context())
					capturedBody, _ = io.ReadAll(r.Body)
					original(w, r)
				}
			}

			server := httptest.NewServer(handler)
			defer server.Close()

			cfg := openai.OpenAIConfig{
				TokenSource: openai.NewAPIKeyTokenSource("sk-test-key"),
				BaseURL:     server.URL,
				Model:       "gpt-4o",
				Client:      server.Client(),
			}
			if name == "Request should use OAuth token source authorization header." {
				cfg.TokenSource = fakeTokenSource{token: "oauth-token"}
			}

			provider, err := openai.NewOpenAI(cfg)
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}

			ctx := context.Background()

			// Special case: test context cancellation.
			if name == "Context cancellation should return error." {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel() // Cancel immediately.
			}

			resp, err := provider.Call(ctx, test.req())

			if test.expErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if test.expErrIs != nil && !errors.Is(err, test.expErrIs) {
					t.Errorf("expected error to wrap %v, got: %v", test.expErrIs, err)
				}
				return
			}

			// Context cancellation test.
			if name == "Context cancellation should return error." {
				if err == nil {
					t.Fatal("expected error from cancelled context, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if test.assert != nil {
				test.assert(t, resp)
			}

			if test.assertRequest != nil {
				test.assertRequest(t, capturedReq, capturedBody)
			}
		})
	}
}

func TestOpenAIProviderModelInfo(t *testing.T) {
	provider, err := openai.NewOpenAI(openai.OpenAIConfig{
		TokenSource: openai.NewAPIKeyTokenSource("sk-test"),
		Model:       "gpt-4o",
	})
	require.NoError(t, err)

	info := provider.ModelInfo()
	assert.Equal(t, "gpt-4o", info.ID)
	assert.Greater(t, info.ContextWindow, 0)
	assert.Greater(t, info.MaxOutputTokens, 0)
}

type fakeTokenSource struct {
	token string
}

func (f fakeTokenSource) Token(_ context.Context) (string, error) {
	return f.token, nil
}

// --- Test helpers ---

// jsonHandler creates an HTTP handler that responds with the given status and JSON body.
func jsonHandler(status int, body any) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(body)
	}
}
