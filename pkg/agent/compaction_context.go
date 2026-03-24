package agent

import "github.com/slok/gosimov/pkg/model"

// effectiveCompactionContext returns the runtime context derived from full
// append-only history.
//
// If there is a valid latest compaction checkpoint, runtime context shape is:
//   - latest checkpoint message
//   - all messages from checkpoint.Compaction.FirstKeptID onward
//
// If there is no valid latest checkpoint, full history is returned as-is.
func effectiveCompactionContext(messages []model.Message) []model.Message {
	checkpointIdx, checkpoint := latestCompactionCheckpoint(messages)
	if checkpoint == nil || checkpoint.Compaction == nil || checkpoint.Compaction.FirstKeptID == "" {
		return messages
	}

	firstKeptIdx := messageIndexByID(messages, checkpoint.Compaction.FirstKeptID)
	if firstKeptIdx == -1 {
		return messages
	}

	result := make([]model.Message, 0, 1+len(messages)-firstKeptIdx)
	result = append(result, messages[checkpointIdx])
	for i := firstKeptIdx; i < len(messages); i++ {
		if i == checkpointIdx {
			continue
		}
		result = append(result, messages[i])
	}

	return result
}

func latestCompactionCheckpoint(messages []model.Message) (int, *model.Message) {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Kind != model.MessageKindCompaction || messages[i].Compaction == nil {
			continue
		}

		if messages[i].Compaction.FirstKeptID == "" {
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
