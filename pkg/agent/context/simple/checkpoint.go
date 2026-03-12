package simple

import (
	"time"

	"github.com/slok/gosimov/internal/utils/id"
	"github.com/slok/gosimov/pkg/conventions"
	"github.com/slok/gosimov/pkg/model"
)

// applyLatestCheckpoint applies the newest valid checkpoint in the history.
//
// Result shape is:
//   - latest checkpoint message
//   - all messages from FirstKeptID onward
//
// If no valid checkpoint exists, all messages are returned as-is.
func applyLatestCheckpoint(messages []model.Message) []model.Message {
	checkpointIdx, checkpoint := latestCheckpoint(messages)
	if checkpoint == nil || checkpoint.Compaction == nil || checkpoint.Compaction.FirstKeptID == "" {
		return copyMessages(messages)
	}

	firstKeptIdx := messageIndexByID(messages, checkpoint.Compaction.FirstKeptID)
	if firstKeptIdx == -1 {
		return copyMessages(messages)
	}

	return checkpointAndFollowing(messages, checkpointIdx, firstKeptIdx)
}

func messageIndexByID(messages []model.Message, id string) int {
	// Resolve the checkpoint boundary by ID, not by index, so persisted/reloaded
	// histories keep working even if messages were reordered elsewhere.
	for i := range messages {
		if messages[i].ID == id {
			return i
		}
	}

	return -1
}

func checkpointAndFollowing(messages []model.Message, checkpointIdx, firstKeptIdx int) []model.Message {
	result := make([]model.Message, 0, 1+len(messages)-firstKeptIdx)
	result = append(result, messages[checkpointIdx])
	// Keep everything from FirstKeptID onward, skipping a duplicate if the
	// checkpoint itself happens to fall in that range.
	for i := firstKeptIdx; i < len(messages); i++ {
		if i == checkpointIdx {
			continue
		}
		result = append(result, messages[i])
	}

	return result
}

// latestCheckpoint returns the newest compaction checkpoint that has a valid boundary.
func latestCheckpoint(messages []model.Message) (int, *model.Message) {
	// Scan backwards so we find the most recent checkpoint first.
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

// latestSummaryText returns the summary text from the latest checkpoint.
func latestSummaryText(messages []model.Message) string {
	_, checkpoint := latestCheckpoint(messages)
	if checkpoint == nil {
		return ""
	}

	return firstText(*checkpoint)
}

func createCheckpoint(summary, firstKeptID string, tokensBefore int) model.Message {
	return model.Message{
		ID:        id.NewULID(conventions.IDPrefixCompaction),
		Kind:      model.MessageKindCompaction,
		CreatedAt: time.Now(),
		Content: []model.ContentPart{{
			Type: model.ContentPartTypeText,
			Text: summary,
		}},
		Compaction: &model.CompactionData{
			FirstKeptID:  firstKeptID,
			TokensBefore: tokensBefore,
		},
	}
}

func prependCheckpoint(messages []model.Message, checkpoint model.Message) []model.Message {
	result := make([]model.Message, 0, 1+len(messages))
	result = append(result, checkpoint)

	return append(result, messages...)
}

// copyMessages returns a shallow copy of the message slice.
func copyMessages(messages []model.Message) []model.Message {
	if len(messages) == 0 {
		return nil
	}

	result := make([]model.Message, len(messages))
	copy(result, messages)

	return result
}

// firstMessageID returns the first non-empty message ID from a slice.
func firstMessageID(messages []model.Message) string {
	for _, msg := range messages {
		if msg.ID != "" {
			return msg.ID
		}
	}

	return ""
}
