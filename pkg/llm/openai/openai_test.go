package openai_test

import (
	"context"
	"encoding/json"
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
			assert := assert.New(t)

			_, err := openai.NewOpenAI(test.cfg)

			if test.expErr {
				assert.Error(err)
				return
			}

			assert.NoError(err)
		})
	}
}

func TestProviderCall(t *testing.T) {
	tests := map[string]struct {
		// tokenSource overrides the default API key token source when set.
		tokenSource openai.TokenSource
		// serverHandler is the HTTP handler for the mock server.
		serverHandler http.HandlerFunc
		// req is the LLM request to send.
		req func() gllm.Request
		// cancelCtx when true, the context is cancelled before calling the provider.
		cancelCtx bool
		// expErr is true if Call should return an error.
		expErr bool
		// expErrIs is the sentinel error to check with errors.Is.
		expErrIs error
		// assertResp runs custom assertions on the response.
		assertResp func(t *testing.T, resp *gllm.Response)
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
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				assert.Equal(model.MessageKindLLM, resp.Message.Kind)
				require.Len(resp.Message.Content, 1)
				assert.Equal("Hello! How can I help?", resp.Message.Content[0].Text)

				require.NotNil(resp.Message.Metadata)
				assert.Equal(model.StopReasonComplete, resp.Message.Metadata.StopReason)
				assert.Equal("glm-5-free", resp.Message.Metadata.Model)
				assert.Equal("openai", resp.Message.Metadata.Provider)

				require.NotNil(resp.Message.Metadata.Usage)
				assert.Equal(15, resp.Message.Metadata.Usage.InputTokens)
				assert.Equal(8, resp.Message.Metadata.Usage.OutputTokens)
				assert.Equal(23, resp.Message.Metadata.Usage.TotalTokens)
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
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(resp.Message.Metadata)
				require.NotNil(resp.Message.Metadata.Usage)
				assert.Equal(14, resp.Message.Metadata.Usage.InputTokens)
				assert.Equal(9, resp.Message.Metadata.Usage.OutputTokens)
				assert.Equal(6, resp.Message.Metadata.Usage.CacheReadTokens)
				assert.Equal(29, resp.Message.Metadata.Usage.TotalTokens)
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
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				assert.Equal(model.StopReasonToolUse, resp.Message.Metadata.StopReason)
				require.Len(resp.Message.ToolCallRequests, 1)

				tc := resp.Message.ToolCallRequests[0]
				assert.Equal("call_abc123", tc.ID)
				assert.Equal("read", tc.ToolID)

				var args map[string]string
				require.NoError(json.Unmarshal(tc.Arguments, &args))
				assert.Equal("main.go", args["path"])
			},
		},

		"Tool call response should force tool_use stop reason regardless of finish_reason.": {
			serverHandler: jsonHandler(200, map[string]any{
				"model": "gpt-4o",
				"choices": []map[string]any{{
					"message": map[string]any{
						"role": "assistant",
						"tool_calls": []map[string]any{{
							"id":   "call_force_tool_use",
							"type": "function",
							"function": map[string]any{
								"name":      "read",
								"arguments": `{"path":"README.md"}`,
							},
						}},
					},
					"finish_reason": "stop",
				}},
			}),
			req: func() gllm.Request {
				return gllm.Request{
					Messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "read README.md"}}},
					},
				}
			},
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(resp.Message.Metadata)
				assert.Equal(model.StopReasonToolUse, resp.Message.Metadata.StopReason)
				require.Len(resp.Message.ToolCallRequests, 1)
				assert.Equal("call_force_tool_use", resp.Message.ToolCallRequests[0].ID)
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
				assert := assert.New(t)
				require := require.New(t)

				assert.Equal("Bearer sk-test-key", httpReq.Header.Get("Authorization"))
				assert.Equal("application/json", httpReq.Header.Get("Content-Type"))

				var reqBody map[string]json.RawMessage
				require.NoError(json.Unmarshal(body, &reqBody))

				var reqModel string
				require.NoError(json.Unmarshal(reqBody["model"], &reqModel))
				assert.Equal("gpt-4o", reqModel)
			},
		},

		"Request should use OAuth token source authorization header.": {
			tokenSource: fakeTokenSource{token: "oauth-token"},
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
				assert := assert.New(t)
				assert.Equal("Bearer oauth-token", httpReq.Header.Get("Authorization"))
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
				assert := assert.New(t)
				require := require.New(t)

				var reqBody map[string]json.RawMessage
				require.NoError(json.Unmarshal(body, &reqBody))

				var maxTokens int
				require.NoError(json.Unmarshal(reqBody["max_tokens"], &maxTokens))
				assert.Equal(123, maxTokens)
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
				assert := assert.New(t)
				require := require.New(t)

				var reqBody map[string]json.RawMessage
				require.NoError(json.Unmarshal(body, &reqBody))

				var promptCacheKey string
				require.NoError(json.Unmarshal(reqBody["prompt_cache_key"], &promptCacheKey))
				assert.Equal("gosimov-sess-s_abc123", promptCacheKey)
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
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal(model.MessageKindLLM, resp.Message.Kind)
				assert.Empty(resp.Message.Content)
			},
		},

		"Context cancellation should return error.": {
			cancelCtx: true,
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
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			var (
				capturedReq  *http.Request
				capturedBody []byte
			)

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

			tokenSource := openai.TokenSource(openai.NewAPIKeyTokenSource("sk-test-key"))
			if test.tokenSource != nil {
				tokenSource = test.tokenSource
			}

			provider, err := openai.NewOpenAI(openai.OpenAIConfig{
				TokenSource: tokenSource,
				BaseURL:     server.URL,
				Model:       "gpt-4o",
				Client:      server.Client(),
			})
			require.NoError(err)

			ctx := context.Background()
			if test.cancelCtx {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}

			resp, err := provider.Call(ctx, test.req())

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				return
			}

			require.NoError(err)

			if test.assertResp != nil {
				test.assertResp(t, resp)
			}

			if test.assertRequest != nil {
				test.assertRequest(t, capturedReq, capturedBody)
			}
		})
	}
}

func TestOpenAIProviderModelInfo(t *testing.T) {
	tests := map[string]struct {
		model string
	}{
		"Should return model info with correct ID and positive values.": {
			model: "gpt-4o",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			provider, err := openai.NewOpenAI(openai.OpenAIConfig{
				TokenSource: openai.NewAPIKeyTokenSource("sk-test"),
				Model:       test.model,
			})
			require.NoError(err)

			info := provider.ModelInfo()
			assert.Equal(test.model, info.ID)
			assert.Greater(info.ContextWindow, 0)
			assert.Greater(info.MaxOutputTokens, 0)
		})
	}
}

type fakeTokenSource struct {
	token string
}

func (f fakeTokenSource) Token(_ context.Context) (string, error) {
	return f.token, nil
}

// jsonHandler creates an HTTP handler that responds with the given status and JSON body.
func jsonHandler(status int, body any) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(body)
	}
}
