package simple

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/model"
)

// serializeMessages converts messages into a text transcript for summarization.
// We intentionally avoid sending these as regular conversation messages because
// text transcripts make it clearer to the summarizer model that it must summarize,
// not continue the dialogue.
func serializeMessages(messages []model.Message) string {
	parts := make([]string, 0, len(messages))
	for _, msg := range messages {
		s := serializeMessage(msg)
		if s != "" {
			parts = append(parts, s)
		}
	}

	return strings.Join(parts, "\n\n")
}

// serializeMessage renders one message in transcript format.
func serializeMessage(msg model.Message) string {
	switch msg.Kind {
	case model.MessageKindUser:
		return "[User]: " + joinText(msg.Content)

	case model.MessageKindLLM:
		items := make([]string, 0, 2)
		text := joinText(msg.Content)
		if text != "" {
			items = append(items, "[LLM]: "+text)
		}

		if len(msg.ToolCallRequests) > 0 {
			calls := make([]string, 0, len(msg.ToolCallRequests))
			for _, tc := range msg.ToolCallRequests {
				calls = append(calls, fmt.Sprintf("%s(%s)", tc.ToolID, singleLineJSON(tc.Arguments)))
			}
			items = append(items, "[LLM tool calls]: "+strings.Join(calls, "; "))
		}

		return strings.Join(items, "\n")

	case model.MessageKindToolResult:
		tag := "[Tool Result]: "
		if msg.IsError {
			tag = "[Tool Result Error]: "
		}
		return tag + joinText(msg.Content)

	case model.MessageKindCompaction:
		return "[Compaction Summary]: " + joinText(msg.Content)

	default:
		return ""
	}
}

// singleLineJSON normalizes JSON arguments to one-line stable representation.
func singleLineJSON(raw []byte) string {
	if len(raw) == 0 {
		return ""
	}

	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return string(raw)
	}

	b, err := json.Marshal(v)
	if err != nil {
		return string(raw)
	}

	return string(b)
}

// joinText joins all non-empty text content parts in a message.
func joinText(parts []model.ContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}

	return strings.Join(texts, "\n")
}

// firstText returns the first non-empty text part from a message.
func firstText(msg model.Message) string {
	for _, p := range msg.Content {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			return p.Text
		}
	}

	return ""
}

// filterFromLatestCompactionMessage applies the newest valid checkpoint in the history.
//
// Result shape is:
//   - latest checkpoint (compaction) message
//   - all messages from FirstKeptID onward
//
// If no valid checkpoint exists, all messages are returned as-is.
func filterFromLatestCompactionMessage(messages []model.Message) []model.Message {
	compactedMsgIdx, compactedMsg := latestCompactionMsg(messages)
	if compactedMsg == nil || compactedMsg.Compaction == nil || compactedMsg.Compaction.FirstKeptID == "" {
		return messages
	}

	firstKeptIdx := messageIndexByID(messages, compactedMsg.Compaction.FirstKeptID)
	if firstKeptIdx == -1 {
		return messages
	}

	return checkpointAndFollowing(messages, compactedMsgIdx, firstKeptIdx)
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

// latestCompactionMsg returns the newest compaction checkpoint that has a valid boundary.
func latestCompactionMsg(messages []model.Message) (int, *model.Message) {
	// Scan backwards so we find the most recent checkpoint first.
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Kind != model.MessageKindCompaction || msg.Compaction == nil || msg.Compaction.FirstKeptID == "" {
			continue
		}
		return i, &messages[i]
	}

	return -1, nil
}

// latestSummaryText returns the summary text from the latest compaction.
func latestSummaryText(messages []model.Message) string {
	_, checkpoint := latestCompactionMsg(messages)
	if checkpoint == nil {
		return ""
	}

	return firstText(*checkpoint)
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

// contextWindowTokensFromCtx extracts the context window tokens from the provider info in the context
// as the upper layer models could change dinamically on every call, if not present falling back to a default.
func contextWindowTokensFromCtx(ctx context.Context) int {
	info := agent.LLMModelInfoFromCtx(ctx)
	if info != nil && info.ContextWindow > 0 {
		return info.ContextWindow
	}

	return defaultContextWindowTokens
}

// findCutIndex finds the first kept message index using a backwards token window.
//
// Algorithm:
//  1. Walk from newest to oldest accumulating estimated tokens.
//  2. Stop once keepRecentTokens is reached.
//  3. If the cut lands on a tool result, walk backwards until a non-tool-result
//     message so we don't split tool call/result context.
func findCutIndex(messages []model.Message, keepRecentTokens int) int {
	if len(messages) == 0 {
		return -1
	}

	tokens := 0
	cut := -1
	// Walk backwards to keep the most recent context under the target window.
	for i := len(messages) - 1; i >= 0; i-- {
		tokens += estimateMessageTokens(messages[i])
		if tokens >= keepRecentTokens {
			cut = i
			break
		}
	}

	if cut <= 0 {
		return cut
	}

	// Never start the kept window at a tool result.
	// This avoids dangling tool outputs without the assistant request that led to it.
	for cut > 0 && messages[cut].Kind == model.MessageKindToolResult {
		cut--
	}

	return cut
}

// estimateMessagesTokens sums approximate tokens for a message slice.
func estimateMessagesTokens(messages []model.Message) int {
	total := 0
	for _, msg := range messages {
		total += estimateMessageTokens(msg)
	}

	return total
}

// estimateMessageTokens approximates token count with chars/4, Pi-mono style.
//
// This is intentionally simple and deterministic; later versions can use
// model-aware tokenizers or provider usage metadata.
func estimateMessageTokens(msg model.Message) int {
	chars := 0
	for _, p := range msg.Content {
		if p.Type != model.ContentPartTypeText {
			continue
		}
		chars += len(p.Text)
	}

	for _, tc := range msg.ToolCallRequests {
		chars += len(tc.ToolID)
		chars += len(tc.Arguments)
	}

	tokens := chars / 4
	if chars > 0 && tokens == 0 {
		return 1
	}

	return tokens
}
