package simple

import "github.com/slok/gosimov/pkg/model"

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
