package customopenaicompatible_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/customopenaicompatible"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		cfg    customopenaicompatible.Config
		expErr bool
	}{
		"Valid config without auth should succeed.": {
			cfg: customopenaicompatible.Config{BaseURL: "http://example.com/v1", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192},
		},
		"Valid config with token source should succeed.": {
			cfg: customopenaicompatible.Config{BaseURL: "http://example.com/v1", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192, TokenSource: customopenaicompatible.NewAPIKeyTokenSource("secret")},
		},
		"Missing base URL should fail.": {
			cfg:    customopenaicompatible.Config{Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192},
			expErr: true,
		},
		"Missing model should fail.": {
			cfg:    customopenaicompatible.Config{BaseURL: "http://example.com/v1", ContextWindow: 131072, MaxOutputTokens: 8192},
			expErr: true,
		},
		"Missing context window should fail.": {
			cfg:    customopenaicompatible.Config{BaseURL: "http://example.com/v1", Model: "Qwen3.6", MaxOutputTokens: 8192},
			expErr: true,
		},
		"Missing max output tokens should fail.": {
			cfg:    customopenaicompatible.Config{BaseURL: "http://example.com/v1", Model: "Qwen3.6", ContextWindow: 131072},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			_, err := customopenaicompatible.New(test.cfg)

			if test.expErr {
				assert.Error(err)
				assert.ErrorIs(err, pkgerrors.ErrNotValid)
				return
			}

			assert.NoError(err)
		})
	}
}

func TestProviderCall(t *testing.T) {
	tests := map[string]struct {
		cfg           customopenaicompatible.Config
		assertRequest func(t *testing.T, r *http.Request)
	}{
		"No auth should omit authorization header.": {
			cfg: customopenaicompatible.Config{BaseURL: "ignored", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192},
			assertRequest: func(t *testing.T, r *http.Request) {
				t.Helper()
				assert.Equal(t, "/chat/completions", r.URL.Path)
				assert.Empty(t, r.Header.Get("Authorization"))
			},
		},
		"Token source should send bearer header.": {
			cfg: customopenaicompatible.Config{BaseURL: "ignored", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192, TokenSource: customopenaicompatible.NewAPIKeyTokenSource("secret")},
			assertRequest: func(t *testing.T, r *http.Request) {
				t.Helper()
				assert.Equal(t, "Bearer secret", r.Header.Get("Authorization"))
			},
		},
		"Qwen top-level thinking should be sent when configured.": {
			cfg: customopenaicompatible.Config{BaseURL: "ignored", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192, ProviderOptions: customopenaicompatible.NewProviderOptions().WithQwenThinking(false)},
			assertRequest: func(t *testing.T, r *http.Request) {
				t.Helper()
				var body map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				assert.Equal(t, false, body["enable_thinking"])
			},
		},
		"Qwen chat template thinking should be sent when configured.": {
			cfg: customopenaicompatible.Config{BaseURL: "ignored", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192, ProviderOptions: customopenaicompatible.NewProviderOptions().WithQwenChatTemplateThinking(false)},
			assertRequest: func(t *testing.T, r *http.Request) {
				t.Helper()
				var body map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				kwargs, ok := body["chat_template_kwargs"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, false, kwargs["enable_thinking"])
			},
		},
		"Qwen preserve thinking should be sent when configured.": {
			cfg: customopenaicompatible.Config{BaseURL: "ignored", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192, ProviderOptions: customopenaicompatible.NewProviderOptions().WithQwenChatTemplatePreserveThinking(true)},
			assertRequest: func(t *testing.T, r *http.Request) {
				t.Helper()
				var body map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				kwargs, ok := body["chat_template_kwargs"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, true, kwargs["preserve_thinking"])
			},
		},
		"Raw chat template kwargs should be merged with typed Qwen chat template options.": {
			cfg: customopenaicompatible.Config{BaseURL: "ignored", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192, ProviderOptions: customopenaicompatible.NewProviderOptions().WithRawRequestField("chat_template_kwargs", map[string]any{"foo": "bar"}).WithQwenChatTemplateThinking(true).WithQwenChatTemplatePreserveThinking(true)},
			assertRequest: func(t *testing.T, r *http.Request) {
				t.Helper()
				var body map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				kwargs, ok := body["chat_template_kwargs"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, "bar", kwargs["foo"])
				assert.Equal(t, true, kwargs["enable_thinking"])
				assert.Equal(t, true, kwargs["preserve_thinking"])
			},
		},
		"Raw request fields should be sent when configured.": {
			cfg: customopenaicompatible.Config{BaseURL: "ignored", Model: "Qwen3.6", ContextWindow: 131072, MaxOutputTokens: 8192, ProviderOptions: customopenaicompatible.NewProviderOptions().WithRawRequestField("service_tier", "default")},
			assertRequest: func(t *testing.T, r *http.Request) {
				t.Helper()
				var body map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				assert.Equal(t, "default", body["service_tier"])
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if test.assertRequest != nil {
					test.assertRequest(t, r)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"model": "Qwen3.6",
					"choices": []map[string]any{{
						"message":       map[string]any{"role": "assistant", "content": "ok"},
						"finish_reason": "stop",
					}},
				})
			}))
			defer server.Close()

			cfg := test.cfg
			cfg.BaseURL = server.URL
			provider, err := customopenaicompatible.New(cfg)
			require.NoError(err)

			resp, err := provider.Call(context.Background(), llm.Request{Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hello")}}}})
			require.NoError(err)
			require.NotNil(resp.Message.Metadata)
			assert.Equal(t, "custom-openai-compatible", resp.Message.Metadata.Provider)
			assert.Equal(t, "Qwen3.6", resp.Message.Metadata.Model)
			require.Len(resp.Message.Content, 1)
			assert.Equal(t, "ok", resp.Message.Content[0].Text)
			assert.Equal(t, "Qwen3.6", provider.ModelInfo().ID)
			assert.Equal(t, 131072, provider.ModelInfo().ContextWindow)
			assert.Equal(t, 8192, provider.ModelInfo().MaxOutputTokens)
		})
	}
}
