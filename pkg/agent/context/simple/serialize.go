package simple

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/slok/gosimov/pkg/model"
)

// serializeMessages converts messages into a text transcript for summarization.
//
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
				calls = append(calls, fmt.Sprintf("%s(%s)", tc.ToolID, compactJSON(tc.Arguments)))
			}
			items = append(items, "[LLM tool calls]: "+strings.Join(calls, "; "))
		}
		return strings.Join(items, "\n")
	case model.MessageKindToolResult:
		tag := "[Tool Result]"
		if msg.IsError {
			tag = "[Tool Result Error]"
		}
		return tag + ": " + joinText(msg.Content)
	case model.MessageKindCompaction:
		return "[Compaction Summary]: " + joinText(msg.Content)
	default:
		return ""
	}
}

// compactJSON normalizes JSON arguments to one-line stable representation.
func compactJSON(raw []byte) string {
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
