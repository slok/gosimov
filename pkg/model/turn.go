package model

// Turn is a derived grouping of messages representing one conversation turn:
// a user message, followed by an LLM response, followed by any tool results
// produced by that response.
//
// Turns are NOT stored — they are computed from a flat message list via
// [TurnsFromMessages]. The agent loop and UI use turns for structure,
// but storage deals only with individual messages.
type Turn struct {
	Messages []Message
}

// TurnsFromMessages groups a flat message list into turns.
//
// A new turn starts on each user message. All following LLM and tool_result
// messages belong to that turn until the next user message.
//
// Messages that appear before any user message (e.g., an orphan LLM or
// tool_result message) are grouped into their own turn.
func TurnsFromMessages(messages []Message) []Turn {
	if len(messages) == 0 {
		return nil
	}

	var turns []Turn
	var current *Turn

	for _, msg := range messages {
		if msg.Kind == MessageKindUser || current == nil {
			turns = append(turns, Turn{})
			current = &turns[len(turns)-1]
		}

		current.Messages = append(current.Messages, msg)
	}

	return turns
}
