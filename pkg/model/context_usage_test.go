package model_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/pkg/model"
)

func TestContextUsageFromMessages(t *testing.T) {
	tests := map[string]struct {
		messages []model.Message
		exp      model.ContextUsage
	}{
		"Empty messages should return zero ContextUsage.": {
			messages: []model.Message{},
			exp:      model.ContextUsage{},
		},

		"Nil messages should return zero ContextUsage.": {
			messages: nil,
			exp:      model.ContextUsage{},
		},

		"No LLM messages should return zero ContextUsage.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindToolResult},
			},
			exp: model.ContextUsage{},
		},

		"Single LLM message with usage should extract all fields.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{
						InputTokens:      100,
						OutputTokens:     50,
						CacheReadTokens:  30,
						CacheWriteTokens: 20,
						ReasoningTokens:  10,
					},
				}},
			},
			exp: model.ContextUsage{
				InputTokens:      100,
				OutputTokens:     50,
				CacheReadTokens:  30,
				CacheWriteTokens: 20,
				TotalInputTokens: 130,
				ReasoningTokens:  10,
			},
		},

		"Multiple LLM messages should use the last one.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{InputTokens: 100, OutputTokens: 50},
				}},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{InputTokens: 200, OutputTokens: 80, CacheReadTokens: 50},
				}},
			},
			exp: model.ContextUsage{
				InputTokens:      200,
				OutputTokens:     80,
				CacheReadTokens:  50,
				TotalInputTokens: 250,
			},
		},

		"Last LLM with nil Metadata should return zero ContextUsage.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
			},
			exp: model.ContextUsage{},
		},

		"Last LLM with nil Usage should return zero ContextUsage.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonComplete,
				}},
			},
			exp: model.ContextUsage{},
		},

		"Last LLM with nil Usage should not fall back to earlier LLM with usage.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{InputTokens: 100, OutputTokens: 50},
				}},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonComplete,
				}},
			},
			exp: model.ContextUsage{},
		},

		"TotalInputTokens should be InputTokens + CacheReadTokens.": {
			messages: []model.Message{
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{InputTokens: 500, CacheReadTokens: 300},
				}},
			},
			exp: model.ContextUsage{
				InputTokens:      500,
				CacheReadTokens:  300,
				TotalInputTokens: 800,
			},
		},

		"Messages after compaction should use the latest LLM.": {
			messages: []model.Message{
				{Kind: model.MessageKindCompaction},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{InputTokens: 50, OutputTokens: 20},
				}},
			},
			exp: model.ContextUsage{
				InputTokens:      50,
				OutputTokens:     20,
				TotalInputTokens: 50,
			},
		},

		"LLM after tool results should be picked as the latest.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{InputTokens: 100, OutputTokens: 10},
				}},
				{Kind: model.MessageKindToolResult},
				{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{
					Usage: &model.Usage{InputTokens: 150, OutputTokens: 30, CacheReadTokens: 40},
				}},
			},
			exp: model.ContextUsage{
				InputTokens:      150,
				OutputTokens:     30,
				CacheReadTokens:  40,
				TotalInputTokens: 190,
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := model.ContextUsageFromMessages(test.messages)

			assert.Equal(test.exp, got)
		})
	}
}
