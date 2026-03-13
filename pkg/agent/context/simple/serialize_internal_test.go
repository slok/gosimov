package simple

import (
	"encoding/json"
	"testing"

	"github.com/slok/gosimov/pkg/model"
	"github.com/stretchr/testify/assert"
)

func TestSerializeMessage(t *testing.T) {
	tests := map[string]struct {
		msg model.Message
		exp string
	}{
		"User message should serialize with User tag.": {
			msg: model.Message{Kind: model.MessageKindUser, Content: testTextParts("hello")},
			exp: "[User]: hello",
		},
		"LLM text and tool calls should serialize both blocks.": {
			msg: model.Message{
				Kind:    model.MessageKindLLM,
				Content: testTextParts("planning"),
				ToolCallRequests: []model.ToolCallRequest{{
					ID:        "tc1",
					ToolID:    "read",
					Arguments: json.RawMessage(`{ "path" : "main.go" }`),
				}},
			},
			exp: "[LLM]: planning\n[LLM tool calls]: read({\"path\":\"main.go\"})",
		},
		"Tool result should serialize with result tag.": {
			msg: model.Message{Kind: model.MessageKindToolResult, IsError: false, Content: testTextParts("ok")},
			exp: "[Tool Result]: ok",
		},
		"Tool result error should serialize with error tag.": {
			msg: model.Message{Kind: model.MessageKindToolResult, IsError: true, Content: testTextParts("permission denied")},
			exp: "[Tool Result Error]: permission denied",
		},
		"Compaction message should serialize with compaction tag.": {
			msg: model.Message{Kind: model.MessageKindCompaction, Content: testTextParts("summary")},
			exp: "[Compaction Summary]: summary",
		},
		"Unknown kind should serialize as empty.": {
			msg: model.Message{Kind: model.MessageKind("unknown")},
			exp: "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, serializeMessage(test.msg))
		})
	}
}

func TestSerializeMessages(t *testing.T) {
	tests := map[string]struct {
		msgs []model.Message
		exp  string
	}{
		"Multiple serializable messages should be joined with blank lines.": {
			msgs: []model.Message{
				{Kind: model.MessageKindUser, Content: testTextParts("u")},
				{Kind: model.MessageKindLLM, Content: testTextParts("a")},
			},
			exp: "[User]: u\n\n[LLM]: a",
		},
		"Unknown messages should be skipped.": {
			msgs: []model.Message{
				{Kind: model.MessageKind("unknown"), Content: testTextParts("x")},
				{Kind: model.MessageKindUser, Content: testTextParts("u")},
			},
			exp: "[User]: u",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, serializeMessages(test.msgs))
		})
	}
}

func TestCompactJSON(t *testing.T) {
	tests := map[string]struct {
		raw []byte
		exp string
	}{
		"Empty raw should return empty string.": {
			raw: nil,
			exp: "",
		},
		"Valid JSON should be compacted.": {
			raw: []byte(`{ "path" : "main.go", "offset": 10 }`),
			exp: `{"offset":10,"path":"main.go"}`,
		},
		"Invalid JSON should return original string.": {
			raw: []byte(`{"path":"main.go"`),
			exp: `{"path":"main.go"`,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, compactJSON(test.raw))
		})
	}
}

func TestJoinTextAndFirstText(t *testing.T) {
	tests := map[string]struct {
		parts        []model.ContentPart
		expJoin      string
		expFirstText string
	}{
		"Only text parts should be joined.": {
			parts: []model.ContentPart{
				{Type: model.ContentPartTypeText, Text: "a"},
				{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: []byte("x"), MimeType: "image/png"}},
				{Type: model.ContentPartTypeText, Text: "b"},
			},
			expJoin:      "a\nb",
			expFirstText: "a",
		},
		"Empty text parts should be ignored.": {
			parts: []model.ContentPart{
				{Type: model.ContentPartTypeText, Text: ""},
				{Type: model.ContentPartTypeText, Text: "x"},
			},
			expJoin:      "x",
			expFirstText: "x",
		},
		"No text parts should return empty values.": {
			parts:        []model.ContentPart{{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: []byte("x"), MimeType: "image/png"}}},
			expJoin:      "",
			expFirstText: "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.expJoin, joinText(test.parts))
			assert.Equal(t, test.expFirstText, firstText(model.Message{Content: test.parts}))
		})
	}
}

func testTextParts(text string) []model.ContentPart {
	return []model.ContentPart{{Type: model.ContentPartTypeText, Text: text}}
}
