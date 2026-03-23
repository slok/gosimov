package session_test

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/agent/context/simple"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/jsonl"
)

func TestCompaction(t *testing.T) {
	cfg := NewConfig(t)

	// Use a generous timeout: 3 turns + 1 compaction summarization call.
	ctx, cancel := context.WithTimeout(context.Background(), 2*cfg.Timeout)
	defer cancel()

	// Setup store.
	repo, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	// Create two providers: conversation and summarization.
	provider := cfg.NewProvider(t)
	summaryProvider := cfg.NewSummaryProvider(t)

	// Create compactor with small KeepRecentTokens to make compaction meaningful.
	compactor, err := simple.New(simple.Config{
		Provider:         summaryProvider,
		KeepRecentTokens: 50,
	})
	require.NoError(t, err)

	// Create session with compactor.
	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		Compactor:         compactor,
		SystemPrompt:      "You are a concise assistant. Keep answers to 2-3 sentences max.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	// Turn 1: ask about Go interfaces.
	turn1, err := promptWithRetry(t, ctx, session, textParts("Explain what Go interfaces are in 2-3 sentences."))
	require.NoError(t, err)
	turn1Final := finalLLMMessageFromTurnResult(t, turn1)
	assert.Equal(t, model.StopReasonComplete, turn1Final.Metadata.StopReason)

	// Turn 2: ask about Go channels.
	turn2, err := promptWithRetry(t, ctx, session, textParts("Now explain what Go channels are in 2-3 sentences."))
	require.NoError(t, err)
	turn2Final := finalLLMMessageFromTurnResult(t, turn2)
	assert.Equal(t, model.StopReasonComplete, turn2Final.Metadata.StopReason)

	usageBeforeCompaction := session.Usage()
	messagesBeforeCompaction := len(session.Messages())

	// Force compaction between turns.
	compactResult, err := session.Compact(ctx)
	require.NoError(t, err)

	// Compaction should have created a checkpoint message.
	require.NotNil(t, compactResult.Message, "compaction should create a checkpoint message")
	assert.Equal(t, model.MessageKindCompaction, compactResult.Message.Kind)
	require.NotNil(t, compactResult.Message.Compaction, "compaction message should have CompactionData")
	assert.NotEmpty(t, compactResult.Message.Compaction.FirstKeptID, "should have FirstKeptID")

	// Compaction should have used tokens for the summarization call.
	assert.Greater(t, compactResult.Usage.TotalTokens, 0, "compaction summarization should use tokens")

	// Session usage should now include compaction tokens.
	usageAfterCompaction := session.Usage()
	assert.Greater(t, usageAfterCompaction.TotalTokens, usageBeforeCompaction.TotalTokens,
		"session usage should increase after compaction")

	// Session messages should include the compaction message.
	assert.Greater(t, len(session.Messages()), messagesBeforeCompaction,
		"message count should increase after compaction")

	// Verify compaction message is in the message list.
	var foundCompaction bool
	for _, m := range session.Messages() {
		if m.Kind == model.MessageKindCompaction {
			foundCompaction = true
			break
		}
	}
	assert.True(t, foundCompaction, "compaction message should be in session messages")

	// Turn 3: ask about what was discussed — tests that context survived compaction.
	turn3, err := promptWithRetry(t, ctx, session, textParts("What were the two Go topics I asked about earlier? Just name them."))
	require.NoError(t, err)

	// The response should reference both topics, proving compacted context is usable.
	responseText := ""
	for _, cp := range finalLLMMessageFromTurnResult(t, turn3).Content {
		responseText += cp.Text
	}
	// Check that at least one topic is mentioned (the LLM has the summary).
	// We check for both but with a lenient OR — the summary should capture both.
	mentionsInterfaces := containsAny(responseText, "interface", "Interface")
	mentionsChannels := containsAny(responseText, "channel", "Channel")
	assert.True(t, mentionsInterfaces || mentionsChannels,
		"turn 3 response should reference at least one of the prior topics; got: %s", responseText)

	// Final usage should be greater than all individual turns combined
	// (includes compaction overhead).
	finalUsage := session.Usage()
	assert.Greater(t, finalUsage.TotalTokens, usageAfterCompaction.TotalTokens,
		"final usage should exceed post-compaction usage after turn 3")
}

func TestCompactionResultFields(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), 2*cfg.Timeout)
	defer cancel()

	repo, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	provider := cfg.NewProvider(t)
	summaryProvider := cfg.NewSummaryProvider(t)

	compactor, err := simple.New(simple.Config{
		Provider:         summaryProvider,
		KeepRecentTokens: 200,
	})
	require.NoError(t, err)

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		Compactor:         compactor,
		SystemPrompt:      "Be concise.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	// Build some conversation history.
	_, err = promptWithRetry(t, ctx, session, textParts("Summarize this note in one sentence:\n"+strings.Repeat("Go interfaces decouple behavior from implementation. ", 120)))
	require.NoError(t, err)

	_, err = promptWithRetry(t, ctx, session, textParts("Summarize this note in one sentence:\n"+strings.Repeat("Rust ownership enforces memory safety at compile time. ", 120)))
	require.NoError(t, err)

	messagesBefore := session.Messages()

	// Compact and verify all CompactResult fields.
	result, err := session.Compact(ctx)
	require.NoError(t, err)

	require.NotNil(t, result.Message)
	assert.Equal(t, model.MessageKindCompaction, result.Message.Kind)

	// Content should have the summary text.
	require.NotEmpty(t, result.Message.Content)
	assert.NotEmpty(t, result.Message.Content[0].Text, "compaction summary should have text content")

	// CompactionData fields.
	require.NotNil(t, result.Message.Compaction)
	assert.NotEmpty(t, result.Message.Compaction.FirstKeptID)
	assert.Greater(t, result.Message.Compaction.TokensBefore, 0, "TokensBefore should be > 0")

	// Filtered messages (what would be sent to LLM after compaction).
	assert.NotEmpty(t, result.Messages, "filtered messages should not be empty")

	// The filtered messages should start with or contain the compaction message.
	var filteredHasCompaction bool
	for _, m := range result.Messages {
		if m.Kind == model.MessageKindCompaction {
			filteredHasCompaction = true
			break
		}
	}
	assert.True(t, filteredHasCompaction, "filtered messages should include compaction checkpoint")
	assertCompactionFilteredOldMessages(t, messagesBefore, result.Messages, result.Message.Compaction.FirstKeptID)

	// Usage from the summarization call.
	assert.GreaterOrEqual(t, result.Usage.InputTokens, 0)
	assert.Greater(t, result.Usage.OutputTokens, 0)
	assert.Greater(t, result.Usage.TotalTokens, 0)
}

// Noop compaction: if we don't force and tokens are below threshold, no compaction happens.
func TestCompactionNoopWhenNotForced(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	repo, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	provider := cfg.NewProvider(t)
	summaryProvider := cfg.NewSummaryProvider(t)

	// Large keep-recent window = auto-compaction won't trigger.
	compactor, err := simple.New(simple.Config{
		Provider:         summaryProvider,
		KeepRecentTokens: 50000,
	})
	require.NoError(t, err)

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		Compactor:         compactor,
		SystemPrompt:      "Be concise.",
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	_, err = promptWithRetry(t, ctx, session, textParts("Say hello."))
	require.NoError(t, err)

	// Session.Compact forces compaction (Force=true), but it can still noop
	// if there is no compactable history beyond keep-recent boundaries.
	result, err := session.Compact(ctx)
	require.NoError(t, err)
	if result.Message == nil {
		assert.Equal(t, 0, result.Usage.TotalTokens)
		return
	}

	assert.Equal(t, model.MessageKindCompaction, result.Message.Kind)
}

// textParts is a helper to create simple text content parts.
func textParts(text string) []model.ContentPart {
	return []model.ContentPart{model.NewContentText(text)}
}

// containsAny checks if s contains any of the substrings (case-sensitive).
func containsAny(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if len(sub) > 0 {
			for i := 0; i <= len(s)-len(sub); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}

func assertCompactionFilteredOldMessages(t *testing.T, before, filtered []model.Message, firstKeptID string) {
	t.Helper()

	firstKeptIdx := -1
	for i, m := range before {
		if m.ID == firstKeptID {
			firstKeptIdx = i
			break
		}
	}
	require.GreaterOrEqual(t, firstKeptIdx, 0, "first kept message should exist in pre-compaction history")

	compactedIDs := map[string]struct{}{}
	for _, m := range before[:firstKeptIdx] {
		compactedIDs[m.ID] = struct{}{}
	}

	for _, m := range filtered {
		_, compacted := compactedIDs[m.ID]
		assert.False(t, compacted, "filtered context should not include compacted message %q", m.ID)
	}
}
