package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/slok/gosimov/internal/utils/id"
	"github.com/slok/gosimov/pkg/conventions"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

func ensureTurnContextActive(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("turn context canceled (%v): %w", err, pkgerrors.ErrAborted)
	}

	return nil
}

// turnResult is what [Session.runTurn] returns after the turn completes.
type turnResult struct {
	// Message is the final LLM response that ended the turn.
	Message model.Message
	// Messages is all new messages generated during the turn (LLM responses + tool results).
	Messages []model.Message
	// LiveMessages is the final live/effective history after the turn ends.
	LiveMessages []model.Message
	// Usage is the aggregated token usage across all LLM calls in the turn.
	Usage model.Usage
}

func toToolDefinitions(tools []tool.Tool) []llm.ToolDefinition {
	if len(tools) == 0 {
		return nil
	}

	defs := make([]llm.ToolDefinition, len(tools))
	for i, t := range tools {
		defs[i] = llm.ToolDefinition{
			ID:          t.ID(),
			Description: t.Description(),
			Schema:      t.Schema(),
		}
	}

	return defs
}

// newToolResultMessage creates a tool result message.
func newToolResultMessage(toolCallID string, content []model.ContentPart, isError bool) model.Message {
	return model.Message{
		ID:         id.NewULID(conventions.IDPrefixToolResult),
		Kind:       model.MessageKindToolResult,
		Content:    content,
		ToolCallID: toolCallID,
		IsError:    isError,
		CreatedAt:  time.Now(),
	}
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
