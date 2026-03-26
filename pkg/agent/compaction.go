package agent

import (
	"context"
	"fmt"
	"time"

	usageutil "github.com/slok/gosimov/internal/utils/usage"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	gosimovlog "github.com/slok/gosimov/pkg/log"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

// Compact forces a context compaction between turns.
//
// It delegates to [Session.runCompaction] with [CompactOptions.Force] set to true.
// If the compactor creates a compaction checkpoint message, it is appended to the
// conversation history and persisted (persistence happens inside [Session.runCompaction]).
//
// Returns the [CompactResult] so the caller can inspect the compaction message
// and usage. If no compactor is configured (NoopCompactor), the result will have
// a nil Message and the full message list.
//
// Cannot be called while a turn is running — returns [ErrSessionBusy].
func (s *Session) Compact(ctx context.Context) (*agentcontext.CompactResult, error) {
	if err := s.beginRun(SessionOperationCompact); err != nil {
		return nil, err
	}
	defer s.endRun()

	logger := s.sessionLogger(SessionOperationCompact)
	logger.Debugf("Starting message compaction...")

	ctx = s.ctxWithRuntimeInfo(ctx)

	s.mu.Lock()
	messages := s.messages
	s.mu.Unlock()

	result, err := s.runCompaction(ctx, messages, agentcontext.CompactOptions{Force: true})
	if err != nil {
		return nil, err
	}

	if result.SummaryMessage != nil {
		updatedMessages := append(messages, *result.SummaryMessage)
		updatedMessages, err = sanitizeLiveCompactionHistory(updatedMessages)
		if err != nil {
			return nil, fmt.Errorf("sanitizing live history after compaction: %w", err)
		}

		s.mu.Lock()
		s.messages = updatedMessages
		s.usage = usageutil.Add(s.usage, result.Usage)
		s.mu.Unlock()

		logger.Debugf("Compaction succeeded, checkpoint message appended to conversation history")
	} else {
		logger.Debugf("No compaction happened, conversation history left unchanged")
	}

	return result, nil
}

// runCompaction executes one compaction pass.
//
// It delegates to the configured [agentcontext.Compactor] and, if a compaction checkpoint
// message is created, validates and persists it via the message repository.
//
// The caller is responsible for updating in-memory state (appending the message
// to session history, aggregating usage).
func (s *Session) runCompaction(ctx context.Context, messages []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
	logger := s.logger.WithValues(gosimovlog.KV{
		"component": "agent.compaction",
	})

	started := time.Now()

	result, err := s.compactor.Compact(ctx, messages, opts)
	if err != nil {
		return nil, fmt.Errorf("compaction failed: %w", err)
	}

	if result.SummaryMessage == nil {
		return result, nil
	}

	if err := validateCompactionSummary(*result.SummaryMessage, messages); err != nil {
		return nil, fmt.Errorf("invalid compaction summary message: %w", err)
	}

	logger.WithValues(gosimovlog.KV{
		"duration_ms":         time.Since(started).Milliseconds(),
		"usage_input_tokens":  result.Usage.InputTokens,
		"usage_output_tokens": result.Usage.OutputTokens,
	}).Infof("Compaction checkpoint created")

	if err := s.messageRepo.StoreMessages(ctx, s.session.ID, []model.Message{*result.SummaryMessage}); err != nil {
		return nil, fmt.Errorf("persisting compaction message: %w", err)
	}

	return result, nil
}

// sanitizeLiveCompactionHistory returns the live/effective history shape used
// by session runtime memory and LLM calls.
//
// If there is no compaction message, history is returned unchanged.
// If there is a latest compaction message, output shape is:
//   - latest compaction message
//   - all messages from compaction.FirstKeptID onward (excluding duplicate compaction)
//
// A malformed latest compaction message fails fast.
func sanitizeLiveCompactionHistory(messages []model.Message) ([]model.Message, error) {
	compactionIdx, compactionMsg := latestCompactionMessage(messages)
	if compactionMsg == nil {
		return messages, nil
	}

	firstKeptIdx := messageIndexByID(messages, compactionMsg.Compaction.FirstKeptID)
	if firstKeptIdx == -1 {
		return nil, fmt.Errorf("invalid compaction checkpoint %q: first kept id %q not found: %w", compactionMsg.ID, compactionMsg.Compaction.FirstKeptID, pkgerrors.ErrNotValid)
	}

	result := make([]model.Message, 0, 1+len(messages)-firstKeptIdx)
	result = append(result, *compactionMsg)
	for i := firstKeptIdx; i < len(messages); i++ {
		if i == compactionIdx {
			continue
		}

		result = append(result, messages[i])
	}

	return result, nil
}

func latestCompactionMessage(messages []model.Message) (int, *model.Message) {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Kind != model.MessageKindCompaction {
			continue
		}

		return i, &messages[i]
	}

	return -1, nil
}

func messageIndexByID(messages []model.Message, id string) int {
	for i := range messages {
		if messages[i].ID == id {
			return i
		}
	}

	return -1
}

func validateCompactionSummary(summary model.Message, messages []model.Message) error {
	if summary.Kind != model.MessageKindCompaction {
		return fmt.Errorf("summary message kind %q is not compaction: %w", summary.Kind, pkgerrors.ErrNotValid)
	}

	if summary.Compaction == nil || summary.Compaction.FirstKeptID == "" {
		return fmt.Errorf("missing first kept id: %w", pkgerrors.ErrNotValid)
	}

	if messageIndexByID(messages, summary.Compaction.FirstKeptID) == -1 {
		return fmt.Errorf("first kept id %q not found in existing messages: %w", summary.Compaction.FirstKeptID, pkgerrors.ErrNotValid)
	}

	return nil
}
