package simple

import (
	"encoding/json"
	"testing"

	"github.com/slok/gosimov/pkg/model"
	"github.com/stretchr/testify/assert"
)

func TestEstimateMessageTokens(t *testing.T) {
	tests := map[string]struct {
		msg model.Message
		exp int
	}{
		"Text token estimation should use chars divided by four.": {
			msg: model.Message{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "12345678"}}},
			exp: 2,
		},
		"Very short text should have minimum of one token.": {
			msg: model.Message{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "a"}}},
			exp: 1,
		},
		"Tool call arguments should be included in token estimate.": {
			msg: model.Message{ToolCallRequests: []model.ToolCallRequest{{ToolID: "read", Arguments: json.RawMessage(`{"path":"main.go"}`)}}},
			exp: 5,
		},
		"Non text content should not increase token estimate.": {
			msg: model.Message{Content: []model.ContentPart{{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: []byte("abcd"), MimeType: "image/png"}}}},
			exp: 0,
		},
		"Empty message should have zero tokens.": {
			msg: model.Message{},
			exp: 0,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, estimateMessageTokens(test.msg))
		})
	}
}

func TestEstimateMessagesTokens(t *testing.T) {
	tests := map[string]struct {
		msgs []model.Message
		exp  int
	}{
		"Empty slice should return zero.": {
			msgs: nil,
			exp:  0,
		},
		"Multiple messages should be summed.": {
			msgs: []model.Message{
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "12345678"}}}, // 2
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "1234"}}},     // 1
			},
			exp: 3,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, estimateMessagesTokens(test.msgs))
		})
	}
}

func TestFindCutIndex(t *testing.T) {
	tests := map[string]struct {
		msgs             []model.Message
		keepRecentTokens int
		exp              int
	}{
		"Empty messages should return -1.": {
			msgs:             nil,
			keepRecentTokens: 10,
			exp:              -1,
		},
		"Target bigger than total should return -1.": {
			msgs: []model.Message{
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "1234"}}},
			},
			keepRecentTokens: 20,
			exp:              -1,
		},
		"Target matching full window should return zero.": {
			msgs: []model.Message{
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "1234"}}},
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "5678"}}},
			},
			keepRecentTokens: 2,
			exp:              0,
		},
		"Target reached in middle should return middle index.": {
			msgs: []model.Message{
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "12345678"}}},
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "12345678"}}},
				{Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "1234"}}},
			},
			keepRecentTokens: 2,
			exp:              1,
		},
		"Cut should move back when landing on tool result.": {
			msgs: []model.Message{
				{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "12345678"}}},
				{Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "x"}}},
				{Kind: model.MessageKindToolResult, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "x"}}},
				{Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "x"}}},
			},
			keepRecentTokens: 2,
			exp:              1,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, findCutIndex(test.msgs, test.keepRecentTokens))
		})
	}
}
