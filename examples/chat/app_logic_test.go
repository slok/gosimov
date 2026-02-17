package main

import (
	"testing"

	"github.com/slok/gosimov/pkg/model"
	"github.com/stretchr/testify/assert"
)

func TestTrimLoadedMessages(t *testing.T) {
	llmWithToolUse := func(ids ...string) model.Message {
		reqs := make([]model.ToolCallRequest, 0, len(ids))
		for _, id := range ids {
			reqs = append(reqs, model.ToolCallRequest{ID: id, ToolID: "read"})
		}

		return model.Message{Kind: model.MessageKindLLM, ToolCallRequests: reqs}
	}

	toolResult := func(id string) model.Message {
		return model.Message{Kind: model.MessageKindToolResult, ToolCallID: id}
	}

	tests := map[string]struct {
		messages []model.Message
		max      int
		expKinds []model.MessageKind
	}{
		"Not exceeding max history should return all messages.": {
			messages: []model.Message{{Kind: model.MessageKindUser}, {Kind: model.MessageKindLLM}},
			max:      10,
			expKinds: []model.MessageKind{model.MessageKindUser, model.MessageKindLLM},
		},
		"Valid tool-use/result blocks should be kept.": {
			messages: []model.Message{{Kind: model.MessageKindUser}, llmWithToolUse("t1"), toolResult("t1"), {Kind: model.MessageKindLLM}},
			max:      3,
			expKinds: []model.MessageKind{model.MessageKindLLM, model.MessageKindToolResult, model.MessageKindLLM},
		},
		"Trimmed window starting on tool result should drop dangling tool results.": {
			messages: []model.Message{{Kind: model.MessageKindUser}, llmWithToolUse("t1", "t2"), toolResult("t1"), toolResult("t2"), {Kind: model.MessageKindLLM}},
			max:      3,
			expKinds: []model.MessageKind{model.MessageKindLLM},
		},
		"All tool results in trimmed window should return empty.": {
			messages: []model.Message{{Kind: model.MessageKindUser}, llmWithToolUse("t1", "t2"), toolResult("t1"), toolResult("t2")},
			max:      2,
			expKinds: []model.MessageKind{},
		},
		"Dangling tool-use at tail should be removed.": {
			messages: []model.Message{{Kind: model.MessageKindUser}, llmWithToolUse("t1")},
			max:      10,
			expKinds: []model.MessageKind{model.MessageKindUser},
		},
		"Tool-use with missing result should be removed.": {
			messages: []model.Message{{Kind: model.MessageKindUser}, llmWithToolUse("t1", "t2"), toolResult("t1"), {Kind: model.MessageKindUser}},
			max:      10,
			expKinds: []model.MessageKind{model.MessageKindUser, model.MessageKindUser},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := trimLoadedMessages(test.messages, test.max)

			gotKinds := make([]model.MessageKind, 0, len(got))
			for _, msg := range got {
				gotKinds = append(gotKinds, msg.Kind)
			}

			assert.Equal(test.expKinds, gotKinds)
		})
	}
}
