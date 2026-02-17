package model_test

import (
	"testing"

	"github.com/slok/gosimov/pkg/model"
)

func TestTurnsFromMessages(t *testing.T) {
	tests := map[string]struct {
		messages         []model.Message
		expTurnCount     int
		expTurnMsgCounts []int // message count per turn
	}{
		"Empty message list should return no turns.": {
			messages:         []model.Message{},
			expTurnCount:     0,
			expTurnMsgCounts: nil,
		},

		"Single user message should produce one turn.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
			},
			expTurnCount:     1,
			expTurnMsgCounts: []int{1},
		},

		"User followed by LLM should produce one turn.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
			},
			expTurnCount:     1,
			expTurnMsgCounts: []int{2},
		},

		"Two user-LLM exchanges should produce two turns.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
			},
			expTurnCount:     2,
			expTurnMsgCounts: []int{2, 2},
		},

		"User followed by LLM with tool calls and results should be one turn.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
				{Kind: model.MessageKindToolResult},
				{Kind: model.MessageKindToolResult},
			},
			expTurnCount:     1,
			expTurnMsgCounts: []int{4},
		},

		"Full cycle with tool calls then new user message should be two turns.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
				{Kind: model.MessageKindToolResult},
				{Kind: model.MessageKindLLM},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
			},
			expTurnCount:     2,
			expTurnMsgCounts: []int{4, 2},
		},

		"Orphan LLM message before any user message should get its own turn.": {
			messages: []model.Message{
				{Kind: model.MessageKindLLM},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
			},
			expTurnCount:     2,
			expTurnMsgCounts: []int{1, 2},
		},

		"Orphan tool result before any user message should get its own turn.": {
			messages: []model.Message{
				{Kind: model.MessageKindToolResult},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindLLM},
			},
			expTurnCount:     2,
			expTurnMsgCounts: []int{1, 2},
		},

		"Only tool results should produce one turn.": {
			messages: []model.Message{
				{Kind: model.MessageKindToolResult},
				{Kind: model.MessageKindToolResult},
			},
			expTurnCount:     1,
			expTurnMsgCounts: []int{2},
		},

		"Multiple user messages in a row should each start a new turn.": {
			messages: []model.Message{
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindUser},
				{Kind: model.MessageKindUser},
			},
			expTurnCount:     3,
			expTurnMsgCounts: []int{1, 1, 1},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := model.TurnsFromMessages(test.messages)

			if len(got) != test.expTurnCount {
				t.Errorf("expected %d turns, got %d", test.expTurnCount, len(got))
				return
			}

			for i, turn := range got {
				if i < len(test.expTurnMsgCounts) && len(turn.Messages) != test.expTurnMsgCounts[i] {
					t.Errorf("turn %d: expected %d messages, got %d", i, test.expTurnMsgCounts[i], len(turn.Messages))
				}
			}
		})
	}
}
