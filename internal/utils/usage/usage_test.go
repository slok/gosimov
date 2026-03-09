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
			raw:                    model.Usage{InputTokens: 100, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, ReasoningTokens: 5},
			inputIncludesCacheRead: true,
			exp:                    model.Usage{InputTokens: 75, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, TotalTokens: 150, ReasoningTokens: 5},
		},

		"Input already non-cached should stay unchanged.": {
			raw: model.Usage{InputTokens: 100, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, ReasoningTokens: 5},
			exp: model.Usage{InputTokens: 100, OutputTokens: 40, CacheReadTokens: 25, CacheWriteTokens: 10, TotalTokens: 175, ReasoningTokens: 5},
		},

		"Negative values should be clamped to zero.": {
			raw: model.Usage{InputTokens: -10, OutputTokens: -1, CacheReadTokens: -3, CacheWriteTokens: -2, ReasoningTokens: -4},
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
			assert := assert.New(t)

			got := Normalize(test.raw, test.inputIncludesCacheRead)
			assert.Equal(test.exp, got)
		})
	}
}

func TestAdd(t *testing.T) {
	tests := map[string]struct {
		a   model.Usage
		b   model.Usage
		exp model.Usage
	}{
		"Usage should add all token fields.": {
			a:   model.Usage{InputTokens: 10, OutputTokens: 5, CacheReadTokens: 3, CacheWriteTokens: 1, ReasoningTokens: 2},
			b:   model.Usage{InputTokens: 20, OutputTokens: 7, CacheReadTokens: 4, CacheWriteTokens: 2, ReasoningTokens: 3},
			exp: model.Usage{InputTokens: 30, OutputTokens: 12, CacheReadTokens: 7, CacheWriteTokens: 3, TotalTokens: 52, ReasoningTokens: 5},
		},

		"Negative values should be ignored when aggregating.": {
			a:   model.Usage{InputTokens: 10},
			b:   model.Usage{InputTokens: -20},
			exp: model.Usage{InputTokens: 10, TotalTokens: 10},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := Add(test.a, test.b)
			assert.Equal(test.exp, got)
		})
	}
}

func TestWithTotal(t *testing.T) {
	tests := map[string]struct {
		input    model.Usage
		expTotal int
	}{
		"Should recalculate total from component tokens.": {
			input:    model.Usage{InputTokens: 11, OutputTokens: 7, CacheReadTokens: 2, CacheWriteTokens: 3, TotalTokens: 999},
			expTotal: 23,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := WithTotal(test.input)
			assert.Equal(test.expTotal, got.TotalTokens)
		})
	}
}
