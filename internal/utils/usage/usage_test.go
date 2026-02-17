package usage

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/pkg/model"
)

func TestNormalize(t *testing.T) {
	tests := map[string]struct {
		raw                    model.Usage
		inputIncludesCacheRead bool
		exp                    model.Usage
	}{
		"Input including cache should be normalized to non-cached input.": {
			raw:                    model.Usage{InputTokens: 100, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, ReasoningTokens: 5, CostUSD: 0.01},
			inputIncludesCacheRead: true,
			exp:                    model.Usage{InputTokens: 75, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, TotalTokens: 150, ReasoningTokens: 5, CostUSD: 0.01},
		},

		"Input already non-cached should stay unchanged.": {
			raw: model.Usage{InputTokens: 100, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, ReasoningTokens: 5, CostUSD: 0.01},
			exp: model.Usage{InputTokens: 100, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, TotalTokens: 175, ReasoningTokens: 5, CostUSD: 0.01},
		},

		"Negative values should be clamped to zero.": {
			raw: model.Usage{InputTokens: -10, OutputTokens: -1, CacheReadTokens: -3, CacheWriteTokens: -2, ReasoningTokens: -4, CostUSD: -0.5},
			exp: model.Usage{},
		},

		"Input should not go below zero when cache is larger than input.": {
			raw:                    model.Usage{InputTokens: 10, CacheReadTokens: 30, OutputTokens: 1},
			inputIncludesCacheRead: true,
			exp:                    model.Usage{OutputTokens: 1, CacheReadTokens: 30, TotalTokens: 31},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := Normalize(test.raw, test.inputIncludesCacheRead)
			assert.Equal(t, test.exp, got)
		})
	}
}

func TestAdd(t *testing.T) {
	tests := map[string]struct {
		a   model.Usage
		b   model.Usage
		exp model.Usage
	}{
		"Usage should add all token fields and cost.": {
			a:   model.Usage{InputTokens: 10, OutputTokens: 5, CacheReadTokens: 3, CacheWriteTokens: 1, ReasoningTokens: 2, CostUSD: 0.1},
			b:   model.Usage{InputTokens: 20, OutputTokens: 7, CacheReadTokens: 4, CacheWriteTokens: 2, ReasoningTokens: 3, CostUSD: 0.2},
			exp: model.Usage{InputTokens: 30, OutputTokens: 12, CacheReadTokens: 7, CacheWriteTokens: 3, TotalTokens: 52, ReasoningTokens: 5, CostUSD: 0.3},
		},

		"Negative values should be ignored when aggregating.": {
			a:   model.Usage{InputTokens: 10, CostUSD: 0.1},
			b:   model.Usage{InputTokens: -20, CostUSD: -1},
			exp: model.Usage{InputTokens: 10, TotalTokens: 10, CostUSD: 0.1},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := Add(test.a, test.b)
			assert.InDelta(t, test.exp.CostUSD, got.CostUSD, 0.0000001)
			got.CostUSD = test.exp.CostUSD
			assert.Equal(t, test.exp, got)
		})
	}
}

func TestWithTotal(t *testing.T) {
	u := model.Usage{InputTokens: 11, OutputTokens: 7, CacheReadTokens: 2, CacheWriteTokens: 3, TotalTokens: 999}

	got := WithTotal(u)

	assert.Equal(t, 23, got.TotalTokens)
}
