package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
			assert := assert.New(t)

			_, err := openai.NewChatGPT(test.cfg)

			if test.expErr {
				assert.Error(err)
				return
			}

			assert.NoError(err)
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
		assertResp    func(t *testing.T, resp *gllm.Response)
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
					Content: []model.ContentPart{model.NewContentText("hi")},
				}},
			},
			assertRequest: func(t *testing.T, req *http.Request, body []byte) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				assert.Equal("/codex/responses", req.URL.Path)
				assert.Equal("Bearer "+jwtWithAccount("acct-123"), req.Header.Get("Authorization"))
				assert.Equal("acct-123", req.Header.Get("chatgpt-account-id"))
				assert.Equal("responses=experimental", req.Header.Get("OpenAI-Beta"))
				assert.Equal("pi", req.Header.Get("originator"))

				var payload map[string]any
				require.NoError(json.Unmarshal(body, &payload))
				assert.Equal("gpt-5.3-codex", payload["model"])
			},
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(resp.Message.Content, 1)
				assert.Equal("Hello from codex", resp.Message.Content[0].Text)

				require.NotNil(resp.Message.Metadata)
				assert.Equal("openai-codex", resp.Message.Metadata.Provider)
				assert.Equal(model.StopReasonComplete, resp.Message.Metadata.StopReason)

				require.NotNil(resp.Message.Metadata.Usage)
				assert.Equal(8, resp.Message.Metadata.Usage.InputTokens)
				assert.Equal(2, resp.Message.Metadata.Usage.CacheReadTokens)
				assert.Equal(15, resp.Message.Metadata.Usage.TotalTokens)
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
			req: gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("read main.go")}}}},
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(resp.Message.Metadata)
				assert.Equal(model.StopReasonToolUse, resp.Message.Metadata.StopReason)
				require.Len(resp.Message.ToolCallRequests, 1)
				assert.Equal("read", resp.Message.ToolCallRequests[0].ToolID)
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
				Messages:  []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}}},
				Config:    gllm.RequestConfig{EnablePromptCache: true},
			},
			assertRequest: func(t *testing.T, _ *http.Request, body []byte) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				var payload struct {
					PromptCacheKey       string `json:"prompt_cache_key"`
					PromptCacheRetention string `json:"prompt_cache_retention"`
				}
				require.NoError(json.Unmarshal(body, &payload))
				assert.Equal("gosimov-sess-s_123", payload.PromptCacheKey)
				assert.Empty(payload.PromptCacheRetention)
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
			req: gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}}}},
			assertResp: func(t *testing.T, resp *gllm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(resp.Message.Content, 1)
				assert.Equal("SSE works", resp.Message.Content[0].Text)
			},
		},

		"Missing account ID in token should fail.": {
			token:         "abc.def.ghi",
			serverHandler: jsonHandler(200, map[string]any{"model": "gpt-5.3-codex", "status": "completed", "output": []any{}}),
			req:           gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}}}},
			expErr:        true,
			expErrIs:      pkgerrors.ErrNotValid,
		},

		"API error should wrap llm error.": {
			token: jwtWithAccount("acct-123"),
			serverHandler: jsonHandler(429, map[string]any{
				"error": map[string]any{"message": "rate limit", "type": "rate_limit"},
			}),
			req:      gllm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}}}},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

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
			require.NoError(err)

			resp, err := provider.Call(context.Background(), test.req)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				return
			}

			require.NoError(err)

			if test.assertRequest != nil {
				test.assertRequest(t, capturedReq, capturedBody)
			}
			if test.assertResp != nil {
				test.assertResp(t, resp)
			}
		})
	}
}

func TestCodexUsesRequestToolDefinitions(t *testing.T) {
	tests := map[string]struct {
		requests       []gllm.Request
		assertPayloads func(t *testing.T, bodies [][]byte)
	}{
		"Tool descriptions should update between calls.": {
			requests: []gllm.Request{
				requestWithCodexToolDescription("desc-v1"),
				requestWithCodexToolDescription("desc-v2"),
			},
			assertPayloads: func(t *testing.T, bodies [][]byte) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				var firstPayload struct {
					Tools []struct {
						Description string `json:"description"`
					} `json:"tools"`
				}
				var secondPayload struct {
					Tools []struct {
						Description string `json:"description"`
					} `json:"tools"`
				}

				require.NoError(json.Unmarshal(bodies[0], &firstPayload))
				require.NoError(json.Unmarshal(bodies[1], &secondPayload))

				require.Len(firstPayload.Tools, 1)
				require.Len(secondPayload.Tools, 1)
				assert.Equal("desc-v1", firstPayload.Tools[0].Description)
				assert.Equal("desc-v2", secondPayload.Tools[0].Description)
			},
		},

		"Requests without tools should omit tools payload.": {
			requests: []gllm.Request{{
				Messages: []model.Message{{
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{model.NewContentText("hello")},
				}},
			}},
			assertPayloads: func(t *testing.T, bodies [][]byte) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				var payload map[string]json.RawMessage
				require.NoError(json.Unmarshal(bodies[0], &payload))
				assert.NotContains(payload, "tools")
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			bodies := make([][]byte, 0, len(test.requests))
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				require.NoError(err)
				bodies = append(bodies, body)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"model":  "gpt-5.3-codex",
					"status": "completed",
					"output": []map[string]any{{
						"type": "message",
						"role": "assistant",
						"content": []map[string]any{{
							"type": "output_text",
							"text": "ok",
						}},
					}},
				})
			}))
			defer server.Close()

			provider, err := openai.NewChatGPT(openai.ChatGPTConfig{
				TokenSource: fakeTokenSource{token: jwtWithAccount("acct-123")},
				BaseURL:     server.URL,
				Model:       "gpt-5.3-codex",
				Client:      server.Client(),
			})
			require.NoError(err)

			for _, req := range test.requests {
				_, err = provider.Call(context.Background(), req)
				require.NoError(err)
			}

			require.Len(bodies, len(test.requests))
			if test.assertPayloads != nil {
				test.assertPayloads(t, bodies)
			}
		})
	}
}

func requestWithCodexToolDescription(description string) gllm.Request {
	return gllm.Request{
		Messages: []model.Message{{
			Kind:    model.MessageKindUser,
			Content: []model.ContentPart{model.NewContentText("hello")},
		}},
		Tools: []gllm.ToolDefinition{{
			ID:          "skill",
			Description: description,
			Schema:      json.RawMessage(`{"type":"object","properties":{}}`),
		}},
	}
}

func TestChatGPTProviderModelInfo(t *testing.T) {
	tests := map[string]struct {
		model string
	}{
		"Should return model info with correct ID and positive values.": {
			model: "gpt-5.3-codex",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			provider, err := openai.NewChatGPT(openai.ChatGPTConfig{
				TokenSource: fakeTokenSource{token: jwtWithAccount("acct-123")},
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

func jwtWithAccount(accountID string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"` + accountID + `"}}`))
	return strings.Join([]string{header, payload, "sig"}, ".")
}
