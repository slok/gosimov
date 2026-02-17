package model

// Usage tracks token consumption and cost for an LLM call.
type Usage struct {
	InputTokens      int
	OutputTokens     int
	CacheReadTokens  int
	CacheWriteTokens int
	TotalTokens      int
	ReasoningTokens  int
	CostUSD          float64
}
