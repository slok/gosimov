package session_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/jsonl"
)

func TestSessionJSONLExport(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	// Setup JSONL store.
	storeDir := t.TempDir()
	repo, err := jsonl.New(jsonl.Config{Dir: storeDir})
	require.NoError(t, err)

	provider := cfg.NewProvider(t)

	// Create session.
	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      "You are helpful. Be concise.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	sessionID := session.Session().ID
	require.NotEmpty(t, sessionID)

	// Prompt.
	_, err = promptWithRetry(t, ctx, session, []model.ContentPart{model.NewContentText(
		"Name three primary colors. Be brief.")})
	require.NoError(t, err)

	// Verify the JSONL file exists and has the correct structure.
	jsonlPath := filepath.Join(storeDir, sessionID+".jsonl")
	data, err := os.ReadFile(jsonlPath)
	require.NoError(t, err, "JSONL file should exist at %s", jsonlPath)

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	require.GreaterOrEqual(t, len(lines), 3, "should have at least session header + user msg + LLM msg")

	// Line 1: session header.
	var header map[string]any
	require.NoError(t, json.Unmarshal([]byte(lines[0]), &header))
	assert.Equal(t, "session", header["type"])
	assert.Equal(t, sessionID, header["id"])

	// Line 2: user message.
	var userMsg map[string]any
	require.NoError(t, json.Unmarshal([]byte(lines[1]), &userMsg))
	assert.Equal(t, "message", userMsg["type"])
	assert.Equal(t, "user", userMsg["kind"])

	// Last line: LLM message with metadata.
	var llmMsg map[string]any
	require.NoError(t, json.Unmarshal([]byte(lines[len(lines)-1]), &llmMsg))
	assert.Equal(t, "message", llmMsg["type"])
	assert.Equal(t, "llm", llmMsg["kind"])

	metadata, ok := llmMsg["metadata"].(map[string]any)
	require.True(t, ok, "LLM message should have metadata")
	assert.Equal(t, "complete", metadata["stop_reason"])
	assert.NotEmpty(t, metadata["model"], "model should be set in JSONL metadata")
	assert.Equal(t, "opencode-go", metadata["provider"])

	// Metadata should contain usage.
	usageMap, ok := metadata["usage"].(map[string]any)
	require.True(t, ok, "metadata should contain usage")
	assert.GreaterOrEqual(t, numberFromAny(t, usageMap["input_tokens"]), float64(0))
	assert.Greater(t, numberFromAny(t, usageMap["output_tokens"]), float64(0))
	assert.Greater(t, numberFromAny(t, usageMap["total_tokens"]), float64(0))
}

func TestSessionLoadFromJSONL(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	storeDir := t.TempDir()
	repo, err := jsonl.New(jsonl.Config{Dir: storeDir})
	require.NoError(t, err)

	provider := cfg.NewProvider(t)

	// Create and use session.
	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      "Be concise.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	_, err = promptWithRetry(t, ctx, session, []model.ContentPart{model.NewContentText(
		"Say 'hello' and nothing else.")})
	require.NoError(t, err)

	originalID := session.Session().ID
	originalMsgs := session.Messages()
	originalUsage := session.Usage()

	// Load the session from the same JSONL store.
	loaded, err := agent.LoadSession(ctx, agent.LoadSessionConfig{
		SessionID:         originalID,
		Provider:          provider,
		SystemPrompt:      "Be concise.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	// Verify identity.
	assert.Equal(t, originalID, loaded.Session().ID)
	assert.True(t, session.Session().CreatedAt.Equal(loaded.Session().CreatedAt))

	// Verify messages were loaded.
	loadedMsgs := loaded.Messages()
	require.Len(t, loadedMsgs, len(originalMsgs))

	for i := range originalMsgs {
		assert.Equal(t, originalMsgs[i].ID, loadedMsgs[i].ID)
		assert.Equal(t, originalMsgs[i].Kind, loadedMsgs[i].Kind)
	}

	// Verify the loaded session can list the same messages from the repo.
	result, err := repo.ListMessages(ctx, originalID, store.ListOpts{})
	require.NoError(t, err)
	assert.Len(t, result.Items, len(originalMsgs))

	// Verify original session had non-zero usage.
	assert.Greater(t, originalUsage.TotalTokens, 0)
}

func TestTokenUsageAccumulation(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), 2*cfg.Timeout)
	defer cancel()

	repo, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	provider := cfg.NewProvider(t)

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      "Be concise. One sentence max.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	// Turn 1.
	turn1, err := promptWithRetry(t, ctx, session, []model.ContentPart{model.NewContentText(
		"What color is the sky?")})
	require.NoError(t, err)
	turn1Final := finalLLMMessageFromTurnResult(t, turn1)
	require.NotNil(t, turn1Final.Metadata)
	require.NotNil(t, turn1Final.Metadata.Usage)
	assert.Greater(t, turn1Final.Metadata.Usage.TotalTokens, 0)

	usageAfterTurn1 := session.Usage()

	// Turn 2.
	turn2, err := promptWithRetry(t, ctx, session, []model.ContentPart{model.NewContentText(
		"What color is grass?")})
	require.NoError(t, err)
	turn2Final := finalLLMMessageFromTurnResult(t, turn2)
	require.NotNil(t, turn2Final.Metadata)
	require.NotNil(t, turn2Final.Metadata.Usage)
	assert.Greater(t, turn2Final.Metadata.Usage.TotalTokens, 0)

	// Session usage should accumulate across turns.
	finalUsage := session.Usage()
	assert.Greater(t, finalUsage.TotalTokens, usageAfterTurn1.TotalTokens,
		"session total tokens should grow after turn 2")
	assert.GreaterOrEqual(t, finalUsage.InputTokens, usageAfterTurn1.InputTokens,
		"session input tokens should not decrease after turn 2")
	assert.Greater(t, finalUsage.OutputTokens, usageAfterTurn1.OutputTokens,
		"session output tokens should grow after turn 2")

	// Messages should have 4: user1, llm1, user2, llm2.
	msgs := session.Messages()
	require.Len(t, msgs, 4)
	assert.Equal(t, model.MessageKindUser, msgs[0].Kind)
	assert.Equal(t, model.MessageKindLLM, msgs[1].Kind)
	assert.Equal(t, model.MessageKindUser, msgs[2].Kind)
	assert.Equal(t, model.MessageKindLLM, msgs[3].Kind)

	// Each LLM message should have usage in metadata.
	for _, m := range msgs {
		if m.Kind == model.MessageKindLLM {
			require.NotNil(t, m.Metadata, "LLM message should have metadata")
			require.NotNil(t, m.Metadata.Usage, "LLM metadata should have usage")
			assert.Greater(t, m.Metadata.Usage.TotalTokens, 0)
		}
	}

	// Verify JSONL file contains all messages.
	result, err := repo.ListMessages(ctx, session.Session().ID, store.ListOpts{})
	require.NoError(t, err)
	assert.Len(t, result.Items, 4, "JSONL should contain all 4 messages")
}

func numberFromAny(t *testing.T, v any) float64 {
	t.Helper()

	switch n := v.(type) {
	case nil:
		return 0
	case float64:
		return n
	case int:
		return float64(n)
	case int32:
		return float64(n)
	case int64:
		return float64(n)
	default:
		t.Fatalf("unexpected number type %T", v)
		return 0
	}
}
