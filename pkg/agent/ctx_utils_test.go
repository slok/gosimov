package agent

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/pkg/model"
)

func TestCtxWithSessionID(t *testing.T) {
	tests := map[string]struct {
		ctx    context.Context
		setID  string
		assert func(t *testing.T, ctx context.Context)
	}{
		"Should store and retrieve session ID.": {
			ctx:   context.Background(),
			setID: "s-123",
			assert: func(t *testing.T, ctx context.Context) {
				t.Helper()
				assert.Equal(t, "s-123", SessionIDFromCtx(ctx))
			},
		},
		"Missing session ID should return empty string.": {
			ctx: context.Background(),
			assert: func(t *testing.T, ctx context.Context) {
				t.Helper()
				assert.Equal(t, "", SessionIDFromCtx(ctx))
			},
		},
		"Later value should override previous value.": {
			ctx:   ctxWithSessionID(context.Background(), "s-old"),
			setID: "s-new",
			assert: func(t *testing.T, ctx context.Context) {
				t.Helper()
				assert.Equal(t, "s-new", SessionIDFromCtx(ctx))
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := test.ctx
			if test.setID != "" {
				ctx = ctxWithSessionID(ctx, test.setID)
			}

			test.assert(t, ctx)
		})
	}
}

func TestCtxWithLLMModelInfo(t *testing.T) {
	tests := map[string]struct {
		ctx    context.Context
		info   *model.LLMModelInfo
		assert func(t *testing.T, ctx context.Context)
	}{
		"Should store and retrieve model info.": {
			ctx: context.Background(),
			info: &model.LLMModelInfo{
				ID:               "gpt-5",
				ContextWindow:    200000,
				MaxOutputTokens:  8192,
				Reasoning:        true,
				ToolCall:         true,
				InputModalities:  []model.LLMModelInputModality{model.LLMModelInputModalityText},
				OutputModalities: []model.LLMModelOutputModality{model.LLMModelOutputModalityText},
			},
			assert: func(t *testing.T, ctx context.Context) {
				t.Helper()
				got := LLMModelInfoFromCtx(ctx)
				if assert.NotNil(t, got) {
					assert.Equal(t, "gpt-5", got.ID)
					assert.Equal(t, 200000, got.ContextWindow)
					assert.Equal(t, 8192, got.MaxOutputTokens)
				}
			},
		},
		"Missing model info should return nil.": {
			ctx: context.Background(),
			assert: func(t *testing.T, ctx context.Context) {
				t.Helper()
				assert.Nil(t, LLMModelInfoFromCtx(ctx))
			},
		},
		"Nil model info should keep context unchanged.": {
			ctx: ctxWithSessionID(context.Background(), "s-1"),
			assert: func(t *testing.T, ctx context.Context) {
				t.Helper()
				assert.Nil(t, LLMModelInfoFromCtx(ctx))
				assert.Equal(t, "s-1", SessionIDFromCtx(ctx))
			},
		},
		"Stored model info should be copied on set.": {
			ctx:  context.Background(),
			info: &model.LLMModelInfo{ID: "before"},
			assert: func(t *testing.T, ctx context.Context) {
				t.Helper()
				got := LLMModelInfoFromCtx(ctx)
				if assert.NotNil(t, got) {
					assert.Equal(t, "before", got.ID)
				}
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := test.ctx

			if name == "Stored model info should be copied on set." {
				ctx = ctxWithLLMModelInfo(ctx, test.info)
				test.info.ID = "after"
				test.assert(t, ctx)
				return
			}

			ctx = ctxWithLLMModelInfo(ctx, test.info)
			test.assert(t, ctx)
		})
	}
}
