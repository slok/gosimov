package agent

import (
	"fmt"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

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
