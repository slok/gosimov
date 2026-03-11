package opencodego_test

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
	"github.com/slok/gosimov/pkg/llm/opencodego"
	"github.com/slok/gosimov/pkg/model"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		cfg    opencodego.Config
		expErr bool
	}{
		"Valid openai-compatible model should succeed.": {
			cfg: opencodego.Config{TokenSource: opencodego.NewAPIKeyTokenSource("test-key"), Model: opencodego.ModelKimiK25},
		},
		"Valid anthropic-compatible model should succeed.": {
			cfg: opencodego.Config{TokenSource: opencodego.NewAPIKeyTokenSource("test-key"), Model: opencodego.ModelMinimaxM25},
		},
		"Missing token source should fail.": {
			cfg:    opencodego.Config{Model: opencodego.ModelKimiK25},
			expErr: true,
		},
		"Missing model should fail.": {
			cfg:    opencodego.Config{TokenSource: opencodego.NewAPIKeyTokenSource("test-key")},
			expErr: true,
		},
		"Unsupported model should fail.": {
			cfg:    opencodego.Config{TokenSource: opencodego.NewAPIKeyTokenSource("test-key"), Model: "unknown-model"},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			_, err := opencodego.New(test.cfg)

			if test.expErr {
				assert.Error(err)
				return
			}

			assert.NoError(err)
		})
	}
}

func TestProviderRoutesByModelFormat(t *testing.T) {
	tests := map[string]struct {
		model         string
		handler       http.HandlerFunc
		expPath       string
		expAuthHeader string
		assertResp    func(t *testing.T, resp *llm.Response)
	}{
		"OpenAI-compatible model should call chat completions with bearer auth.": {
			model:   opencodego.ModelGlm5,
			expPath: "/chat/completions",
			handler: jsonHandler(200, map[string]any{
				"model": opencodego.ModelGlm5,
				"choices": []map[string]any{{
					"message":       map[string]any{"role": "assistant", "content": "hello from glm"},
					"finish_reason": "stop",
				}},
			}),
			expAuthHeader: "Authorization",
			assertResp: func(t *testing.T, resp *llm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(resp.Message.Metadata)
				assert.Equal("opencode-go", resp.Message.Metadata.Provider)
				assert.Equal(model.StopReasonComplete, resp.Message.Metadata.StopReason)
				require.Len(resp.Message.Content, 1)
				assert.Equal("hello from glm", resp.Message.Content[0].Text)
			},
		},
		"Anthropic-compatible model should call messages with x-api-key auth.": {
			model:   opencodego.ModelMinimaxM25,
			expPath: "/messages",
			handler: jsonHandler(200, map[string]any{
				"model":       opencodego.ModelMinimaxM25,
				"stop_reason": "end_turn",
				"content": []map[string]any{{
					"type": "text",
					"text": "hello from minimax",
				}},
			}),
			expAuthHeader: "x-api-key",
			assertResp: func(t *testing.T, resp *llm.Response) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(resp.Message.Metadata)
				assert.Equal("opencode-go", resp.Message.Metadata.Provider)
				assert.Equal(model.StopReasonComplete, resp.Message.Metadata.StopReason)
				require.Len(resp.Message.Content, 1)
				assert.Equal("hello from minimax", resp.Message.Content[0].Text)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			var capturedReq *http.Request
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capturedReq = r
				test.handler(w, r)
			}))
			defer server.Close()

			provider, err := opencodego.New(opencodego.Config{
				TokenSource: opencodego.NewAPIKeyTokenSource("go-key"),
				BaseURL:     server.URL,
				Model:       test.model,
				Client:      server.Client(),
			})
			require.NoError(err)

			resp, err := provider.Call(context.Background(), llm.Request{
				Messages: []model.Message{{
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}},
				}},
			})
			require.NoError(err)
			require.NotNil(capturedReq)

			assert.Equal(test.expPath, capturedReq.URL.Path)
			switch test.expAuthHeader {
			case "Authorization":
				assert.Equal("Bearer go-key", capturedReq.Header.Get("Authorization"))
			case "x-api-key":
				assert.Equal("go-key", capturedReq.Header.Get("x-api-key"))
			default:
				t.Fatalf("unexpected auth header expectation %q", test.expAuthHeader)
			}

			if test.assertResp != nil {
				test.assertResp(t, resp)
			}
		})
	}
}

func jsonHandler(status int, body any) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(body)
	}
}

func TestProviderModelInfo(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	provider, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource("go-key"),
		Model:       opencodego.ModelKimiK25,
	})
	require.NoError(err)

	info := provider.ModelInfo()
	assert.Equal(opencodego.ModelKimiK25, info.ID)
	assert.Greater(info.ContextWindow, 0)
	assert.Greater(info.MaxOutputTokens, 0)
}

func TestAnthropicRouteReadsToolMessagesEndpointPayload(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		capturedBody, err = io.ReadAll(r.Body)
		require.NoError(err)
		_ = r.Body.Close()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"model":"minimax-m2.5","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer server.Close()

	provider, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource("go-key"),
		BaseURL:     server.URL,
		Model:       opencodego.ModelMinimaxM25,
		Client:      server.Client(),
	})
	require.NoError(err)

	_, err = provider.Call(context.Background(), llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ping"}}}}})
	require.NoError(err)

	var payload map[string]any
	require.NoError(json.Unmarshal(capturedBody, &payload))
	assert.Equal(opencodego.ModelMinimaxM25, payload["model"])
	assert.Contains(payload, "max_tokens")
}

func TestOpenAICompatibleRouteSendsPromptCacheKeyWhenEnabled(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		capturedBody, err = io.ReadAll(r.Body)
		require.NoError(err)
		_ = r.Body.Close()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"model":"glm-5","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	provider, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource("go-key"),
		BaseURL:     server.URL,
		Model:       opencodego.ModelGlm5,
		Client:      server.Client(),
	})
	require.NoError(err)

	_, err = provider.Call(context.Background(), llm.Request{
		SessionID: "sess-cache-1",
		Messages: []model.Message{{
			Kind:    model.MessageKindUser,
			Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ping"}},
		}},
		Config: llm.RequestConfig{EnablePromptCache: true},
	})
	require.NoError(err)

	var payload map[string]any
	require.NoError(json.Unmarshal(capturedBody, &payload))

	cacheKey, ok := payload["prompt_cache_key"].(string)
	require.True(ok)
	assert.Equal("gosimov-sess-sess-cache-1", cacheKey)
}

func TestAnthropicRouteSendsCacheControlWhenEnabled(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		capturedBody, err = io.ReadAll(r.Body)
		require.NoError(err)
		_ = r.Body.Close()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"model":"minimax-m2.5","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer server.Close()

	provider, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource("go-key"),
		BaseURL:     server.URL,
		Model:       opencodego.ModelMinimaxM25,
		Client:      server.Client(),
	})
	require.NoError(err)

	_, err = provider.Call(context.Background(), llm.Request{
		SystemPrompt: "be concise",
		Messages: []model.Message{{
			Kind:    model.MessageKindUser,
			Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ping"}},
		}},
		Config: llm.RequestConfig{EnablePromptCache: true},
	})
	require.NoError(err)

	var payload struct {
		System []struct {
			Type         string `json:"type"`
			Text         string `json:"text"`
			CacheControl struct {
				Type string `json:"type"`
			} `json:"cache_control"`
		} `json:"system"`
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type         string `json:"type"`
				Text         string `json:"text"`
				CacheControl struct {
					Type string `json:"type"`
				} `json:"cache_control"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(json.Unmarshal(capturedBody, &payload))

	require.NotEmpty(payload.System)
	assert.Equal("ephemeral", payload.System[0].CacheControl.Type)

	require.NotEmpty(payload.Messages)
	last := payload.Messages[len(payload.Messages)-1]
	require.NotEmpty(last.Content)
	assert.Equal("ephemeral", last.Content[len(last.Content)-1].CacheControl.Type)
}
