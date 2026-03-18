package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/anthropic"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
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
			assert := assert.New(t)

			_, err := anthropic.NewAnthropic(test.cfg)

			if test.expErr {
				assert.Error(err)
				return
			}

			assert.NoError(err)
		})
	}
}

func TestAnthropicProviderCall(t *testing.T) {
	tests := map[string]struct {
		serverHandler http.HandlerFunc
		req           llm.Request
		expErr        bool
		expErrIs      error
		assertResp    func(t *testing.T, resp *llm.Response)
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
			req: llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}}}},
			assertResp: func(t *testing.T, resp *llm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(resp.Message.Content, 1)
				assert.Equal("Hello from Claude", resp.Message.Content[0].Text)

				require.NotNil(resp.Message.Metadata)
				assert.Equal("anthropic", resp.Message.Metadata.Provider)
				assert.Equal(model.StopReasonComplete, resp.Message.Metadata.StopReason)

				require.NotNil(resp.Message.Metadata.Usage)
				assert.Equal(11, resp.Message.Metadata.Usage.InputTokens)
				assert.Equal(2, resp.Message.Metadata.Usage.CacheReadTokens)
				assert.Equal(3, resp.Message.Metadata.Usage.CacheWriteTokens)
				assert.Equal(23, resp.Message.Metadata.Usage.TotalTokens)
			},
			assertRequest: func(t *testing.T, req *http.Request, body []byte) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				assert.Equal("sk-ant-test", req.Header.Get("x-api-key"))
				assert.NotEmpty(req.Header.Get("anthropic-version"))

				var payload map[string]any
				require.NoError(json.Unmarshal(body, &payload))
				assert.Equal(anthropic.ModelClaudeSonnet46, payload["model"])
				assert.Contains(payload, "max_tokens")
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
			req: llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("read main.go")}}}},
			assertResp: func(t *testing.T, resp *llm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(resp.Message.Metadata)
				assert.Equal(model.StopReasonToolUse, resp.Message.Metadata.StopReason)
				require.Len(resp.Message.ToolCallRequests, 1)
				assert.Equal("read", resp.Message.ToolCallRequests[0].ToolID)
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
					Content: []model.ContentPart{model.NewContentText("hi")},
				}},
				Config: llm.RequestConfig{EnablePromptCache: true},
			},
			assertRequest: func(t *testing.T, _ *http.Request, body []byte) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

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
				require.NoError(json.Unmarshal(body, &payload))

				require.Len(payload.System, 1)
				assert.Equal("ephemeral", payload.System[0].CacheControl.Type)
				assert.Empty(payload.System[0].CacheControl.TTL)

				require.Len(payload.Messages, 1)
				assert.Equal("user", payload.Messages[0].Role)
				require.Len(payload.Messages[0].Content, 1)
				assert.Equal("ephemeral", payload.Messages[0].Content[0].CacheControl.Type)
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
			req:      llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}}}},
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

func TestAnthropicProviderUsesRequestToolDefinitions(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	bodies := make([][]byte, 0, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(err)
		bodies = append(bodies, body)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":       anthropic.ModelClaudeSonnet46,
			"stop_reason": "end_turn",
			"content": []map[string]any{{
				"type": "text",
				"text": "ok",
			}},
		})
	}))
	defer server.Close()

	provider, err := anthropic.NewAnthropic(anthropic.Config{
		TokenSource: anthropic.NewAPIKeyTokenSource("sk-ant-test"),
		BaseURL:     server.URL,
		Model:       anthropic.ModelClaudeSonnet46,
		Client:      server.Client(),
	})
	require.NoError(err)

	requestWithDescription := func(description string) llm.Request {
		return llm.Request{
			Messages: []model.Message{{
				Kind:    model.MessageKindUser,
				Content: []model.ContentPart{model.NewContentText("hello")},
			}},
			Tools: []llm.ToolDefinition{{
				ID:          "read",
				Description: description,
				Schema:      json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
			}},
		}
	}

	_, err = provider.Call(context.Background(), requestWithDescription("desc-v1"))
	require.NoError(err)
	_, err = provider.Call(context.Background(), requestWithDescription("desc-v2"))
	require.NoError(err)

	require.Len(bodies, 2)

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
			assert := assert.New(t)

			_, err := anthropic.NewClaude(test.cfg)

			if test.expErr {
				assert.Error(err)
				return
			}

			assert.NoError(err)
		})
	}
}

func TestClaudeProviderCall(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Should use bearer auth, send Claude headers, normalize tool names, and restore tool IDs.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

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
					Client:      server.Client(),
				})
				require.NoError(err)

				resp, err := provider.Call(context.Background(), llm.Request{
					SystemPrompt: "Keep it short",
					Tools: []llm.ToolDefinition{{
						ID:          "read",
						Description: "Read a file",
						Schema:      json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
					}},
					Messages: []model.Message{{
						Kind:    model.MessageKindUser,
						Content: []model.ContentPart{model.NewContentText("read main.go")},
					}},
				})
				require.NoError(err)

				assert.Equal("Bearer oauth-token", capturedReq.Header.Get("Authorization"))
				assert.NotEmpty(capturedReq.Header.Get("anthropic-beta"))
				assert.Equal("cli", capturedReq.Header.Get("x-app"))

				var payload struct {
					System string `json:"system"`
					Tools  []struct {
						Name string `json:"name"`
					} `json:"tools"`
				}
				require.NoError(json.Unmarshal(capturedBody, &payload))
				assert.NotEmpty(payload.System)
				assert.Contains(payload.System, "Claude Code")
				require.Len(payload.Tools, 1)
				assert.Equal("Read", payload.Tools[0].Name)

				require.Len(resp.Message.ToolCallRequests, 1)
				assert.Equal("read", resp.Message.ToolCallRequests[0].ToolID)
				require.NotNil(resp.Message.Metadata)
				assert.Equal("anthropic-claude", resp.Message.Metadata.Provider)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

func TestAnthropicProviderModelInfo(t *testing.T) {
	tests := map[string]struct {
		newProvider func(t *testing.T) llm.Provider
		expModelID  string
	}{
		"Anthropic provider should return correct model info.": {
			newProvider: func(t *testing.T) llm.Provider {
				t.Helper()
				p, err := anthropic.NewAnthropic(anthropic.Config{
					TokenSource: anthropic.NewAPIKeyTokenSource("sk-ant-test"),
					Model:       anthropic.ModelClaudeSonnet46,
				})
				require.New(t).NoError(err)
				return p
			},
			expModelID: anthropic.ModelClaudeSonnet46,
		},
		"Claude provider should return correct model info.": {
			newProvider: func(t *testing.T) llm.Provider {
				t.Helper()
				p, err := anthropic.NewClaude(anthropic.ClaudeConfig{
					TokenSource: fakeTokenSource{token: "oauth"},
					Model:       anthropic.ModelClaudeSonnet46,
				})
				require.New(t).NoError(err)
				return p
			},
			expModelID: anthropic.ModelClaudeSonnet46,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			provider := test.newProvider(t)
			info := provider.ModelInfo()
			assert.Equal(test.expModelID, info.ID)
			assert.Greater(info.ContextWindow, 0)
		})
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
