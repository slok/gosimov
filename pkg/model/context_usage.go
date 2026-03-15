package model

// ContextUsage represents context window utilization based on the most
// recent LLM call's reported token usage.
//
// Use [ContextUsageFromMessages] to extract this from a conversation history.
// Combine TotalInputTokens with [LLMModelInfo.ContextWindow] to compute
// context window utilization percentage.
type ContextUsage struct {
	// InputTokens is the non-cached input tokens from the last LLM call.
	InputTokens int
	// OutputTokens is the output tokens from the last LLM call.
	OutputTokens int
	// CacheReadTokens is tokens read from provider cache on the last LLM call.
	CacheReadTokens int
	// CacheWriteTokens is tokens written to provider cache on the last LLM call.
	CacheWriteTokens int
	// TotalInputTokens is InputTokens + CacheReadTokens — the actual context
	// size the provider processed. This is the number to compare against
	// [LLMModelInfo.ContextWindow].
	TotalInputTokens int
	// ReasoningTokens is reasoning/chain-of-thought tokens from the last LLM call.
	ReasoningTokens int
}

// ContextUsageFromMessages returns context usage from the most recent LLM
// message in the conversation.
//
// It walks messages backwards and returns the usage from the last
// [MessageKindLLM] message. If that message has no usage metadata,
// a zero [ContextUsage] is returned — it does not fall back to earlier messages.
//
// Returns a zero [ContextUsage] if no LLM message is found.
func ContextUsageFromMessages(messages []Message) ContextUsage {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Kind != MessageKindLLM {
			continue
		}

		if msg.Metadata == nil || msg.Metadata.Usage == nil {
			return ContextUsage{}
		}

		u := msg.Metadata.Usage
		return ContextUsage{
			InputTokens:      u.InputTokens,
			OutputTokens:     u.OutputTokens,
			CacheReadTokens:  u.CacheReadTokens,
			CacheWriteTokens: u.CacheWriteTokens,
			TotalInputTokens: u.InputTokens + u.CacheReadTokens,
			ReasoningTokens:  u.ReasoningTokens,
		}
	}

	return ContextUsage{}
}
