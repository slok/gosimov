package session_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/jsonl"
)

func TestSimpleResponse(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	// Setup store.
	repo, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	// Create provider without tools.
	provider := cfg.NewProvider(t, nil)

	// Create session.
	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      "You are a helpful assistant. Be concise.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	// Prompt with a simple deterministic question.
	result, err := promptWithRetry(t, ctx, session, []model.ContentPart{
		{Type: model.ContentPartTypeText, Text: "What is 2+2? Reply with just the number, nothing else."},
	})
	require.NoError(t, err)

	// The response should be a complete LLM message.
	assert.Equal(t, model.MessageKindLLM, result.Message.Kind)
	require.NotEmpty(t, result.Message.Content, "LLM response should have content")
	assert.Contains(t, result.Message.Content[0].Text, "4")

	// Metadata should be populated.
	require.NotNil(t, result.Message.Metadata, "LLM message should have metadata")
	assert.Equal(t, model.StopReasonComplete, result.Message.Metadata.StopReason)
	assert.Equal(t, "opencode-go", result.Message.Metadata.Provider)
	assert.NotEmpty(t, result.Message.Metadata.Model, "model should be set in metadata")

	// Token usage should be tracked.
	usage := session.Usage()
	assert.Greater(t, usage.TotalTokens, 0, "total tokens should be > 0")
	assert.GreaterOrEqual(t, usage.InputTokens, 0, "input tokens should be >= 0")
	assert.Greater(t, usage.OutputTokens, 0, "output tokens should be > 0")

	// Turn result usage should match session usage (single turn).
	assert.Equal(t, usage.TotalTokens, result.Usage.TotalTokens)

	// Messages should contain the user message + LLM response.
	msgs := session.Messages()
	require.Len(t, msgs, 2)
	assert.Equal(t, model.MessageKindUser, msgs[0].Kind)
	assert.Equal(t, model.MessageKindLLM, msgs[1].Kind)
}
