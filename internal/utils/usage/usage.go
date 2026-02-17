package usage

import "github.com/slok/gosimov/pkg/model"

// Normalize returns canonical usage semantics.
//
// InputTokens is always non-cached input tokens. Set inputIncludesCacheRead=true
// for providers that include cached reads in input token counts.
func Normalize(raw model.Usage, inputIncludesCacheRead bool) model.Usage {
	input := nonNegative(raw.InputTokens)
	output := nonNegative(raw.OutputTokens)
	cacheRead := nonNegative(raw.CacheReadTokens)
	cacheWrite := nonNegative(raw.CacheWriteTokens)
	reasoning := nonNegative(raw.ReasoningTokens)
	cost := raw.CostUSD
	if cost < 0 {
		cost = 0
	}

	if inputIncludesCacheRead {
		input -= cacheRead
		if input < 0 {
			input = 0
		}
	}

	u := model.Usage{
		InputTokens:      input,
		OutputTokens:     output,
		CacheReadTokens:  cacheRead,
		CacheWriteTokens: cacheWrite,
		ReasoningTokens:  reasoning,
		CostUSD:          cost,
	}

	u.TotalTokens = Total(u)

	return u
}

// Add aggregates usage values and keeps canonical totals.
func Add(a model.Usage, b model.Usage) model.Usage {
	u := model.Usage{
		InputTokens:      nonNegative(a.InputTokens) + nonNegative(b.InputTokens),
		OutputTokens:     nonNegative(a.OutputTokens) + nonNegative(b.OutputTokens),
		CacheReadTokens:  nonNegative(a.CacheReadTokens) + nonNegative(b.CacheReadTokens),
		CacheWriteTokens: nonNegative(a.CacheWriteTokens) + nonNegative(b.CacheWriteTokens),
		ReasoningTokens:  nonNegative(a.ReasoningTokens) + nonNegative(b.ReasoningTokens),
		CostUSD:          nonNegativeFloat(a.CostUSD) + nonNegativeFloat(b.CostUSD),
	}

	u.TotalTokens = Total(u)

	return u
}

// WithTotal ensures TotalTokens matches usage components.
func WithTotal(u model.Usage) model.Usage {
	u.TotalTokens = Total(u)
	return u
}

// Total returns total context tokens (input + output + cache read + cache write).
func Total(u model.Usage) int {
	return nonNegative(u.InputTokens) + nonNegative(u.OutputTokens) + nonNegative(u.CacheReadTokens) + nonNegative(u.CacheWriteTokens)
}

func nonNegative(v int) int {
	if v < 0 {
		return 0
	}

	return v
}

func nonNegativeFloat(v float64) float64 {
	if v < 0 {
		return 0
	}

	return v
}
