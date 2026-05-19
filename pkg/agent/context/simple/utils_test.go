package simple

import (
	"encoding/json"
	"testing"

	"github.com/slok/gosimov/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestLatestCompactionMsg(t *testing.T) {
	tests := map[string]struct {
		msgs   []model.Message
		expIdx int
		expID  string
	}{
		"Missing checkpoints should return none.": {
			msgs:   []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
			expIdx: -1,
		},
		"Should skip invalid checkpoints and return latest valid.": {
			msgs: []model.Message{
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{}},
				{ID: "c2", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
				{ID: "c3", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{}},
			},
			expIdx: 1,
			expID:  "c2",
		},
		"Should return newest valid checkpoint.": {
			msgs: []model.Message{
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
				{ID: "c2", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m3"}},
			},
			expIdx: 1,
			expID:  "c2",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			idx, cp := latestCompactionMsg(test.msgs)
			assert.Equal(test.expIdx, idx)
			if test.expIdx == -1 {
				assert.Nil(cp)
				return
			}

			require.NotNil(t, cp)
			assert.Equal(test.expID, cp.ID)
		})
	}
}

func TestFilterFromLatestCompactionMessage(t *testing.T) {
	tests := map[string]struct {
		msgs    []model.Message
		expMsgs []model.Message
	}{
		"No checkpoint should return all messages.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
			expMsgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
		},
		"Unknown first kept id should return all messages.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "missing"}},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
			expMsgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "missing"}},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
		},
		"Should apply latest checkpoint and keep from first kept id onward.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
				{ID: "m2", Kind: model.MessageKindUser},
				{ID: "c2", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m3"}},
				{ID: "m3", Kind: model.MessageKindLLM},
			},
			expMsgs: []model.Message{
				{ID: "c2", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m3"}},
				{ID: "m3", Kind: model.MessageKindLLM},
			},
		},
		"Checkpoint in kept range should not be duplicated.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "m2", Kind: model.MessageKindLLM},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
				{ID: "m3", Kind: model.MessageKindUser},
			},
			expMsgs: []model.Message{
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
				{ID: "m2", Kind: model.MessageKindLLM},
				{ID: "m3", Kind: model.MessageKindUser},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := filterFromLatestCompactionMessage(test.msgs)
			assert.Equal(t, test.expMsgs, got)
		})
	}
}

func TestExtractLatestSummaryText(t *testing.T) {
	tests := map[string]struct {
		msgs       []model.Message
		expText    string
		expRemain  int
	}{
		"No checkpoint should return empty and unchanged messages.": {
			msgs:      []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
			expText:   "",
			expRemain: 1,
		},
		"Latest checkpoint should return first text and remove it from slice.": {
			msgs: []model.Message{
				{ID: "c1", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("old")}, Compaction: &model.CompactionData{FirstKeptID: "m1"}},
				{ID: "c2", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("new")}, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
			},
			expText:   "new",
			expRemain: 1,
		},
		"Single compaction message should return text and empty slice.": {
			msgs: []model.Message{
				{ID: "c1", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("summary")}, Compaction: &model.CompactionData{FirstKeptID: "m1"}},
			},
			expText:   "summary",
			expRemain: 0,
		},
		"Compaction in the middle should be removed correctly.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hello")}},
				{ID: "c1", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("checkpoint")}, Compaction: &model.CompactionData{FirstKeptID: "m0"}},
				{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{model.NewContentText("world")}},
			},
			expText:   "checkpoint",
			expRemain: 2,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			gotText, gotRemain := extractLatestSummaryText(test.msgs)
			assert.Equal(t, test.expText, gotText)
			assert.Equal(t, test.expRemain, len(gotRemain))
		})
	}
}

func TestFirstMessageID(t *testing.T) {
	tests := map[string]struct {
		msgs []model.Message
		exp  string
	}{
		"Missing ids should return empty.": {
			msgs: []model.Message{{ID: ""}, {ID: ""}},
			exp:  "",
		},
		"Should return first non-empty id.": {
			msgs: []model.Message{{ID: ""}, {ID: "m2"}, {ID: "m3"}},
			exp:  "m2",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, firstMessageID(test.msgs))
		})
	}
}

func TestEstimateMessageTokens(t *testing.T) {
	tests := map[string]struct {
		msg model.Message
		exp int
	}{
		"Text token estimation should use chars divided by four.": {
			msg: model.Message{Content: []model.ContentPart{model.NewContentText("12345678")}},
			exp: 2,
		},
		"Very short text should have minimum of one token.": {
			msg: model.Message{Content: []model.ContentPart{model.NewContentText("a")}},
			exp: 1,
		},
		"Tool call arguments should be included in token estimate.": {
			msg: model.Message{ToolCallRequests: []model.ToolCallRequest{{ToolID: "read", Arguments: json.RawMessage(`{"path":"main.go"}`)}}},
			exp: 5,
		},
		"Non text content should not increase token estimate.": {
			msg: model.Message{Content: []model.ContentPart{model.NewContentImage([]byte("abcd"), "image/png")}},
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
				{Content: []model.ContentPart{model.NewContentText("12345678")}},
				{Content: []model.ContentPart{model.NewContentText("1234")}},
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
				{Content: []model.ContentPart{model.NewContentText("1234")}},
			},
			keepRecentTokens: 20,
			exp:              -1,
		},
		"Target matching full window should return zero.": {
			msgs: []model.Message{
				{Content: []model.ContentPart{model.NewContentText("1234")}},
				{Content: []model.ContentPart{model.NewContentText("5678")}},
			},
			keepRecentTokens: 2,
			exp:              0,
		},
		"Target reached in middle should return middle index.": {
			msgs: []model.Message{
				{Content: []model.ContentPart{model.NewContentText("12345678")}},
				{Content: []model.ContentPart{model.NewContentText("12345678")}},
				{Content: []model.ContentPart{model.NewContentText("1234")}},
			},
			keepRecentTokens: 2,
			exp:              1,
		},
		"Cut should move back when landing on tool result.": {
			msgs: []model.Message{
				{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("12345678")}},
				{Kind: model.MessageKindLLM, Content: []model.ContentPart{model.NewContentText("x")}},
				{Kind: model.MessageKindToolResult, Content: []model.ContentPart{model.NewContentText("x")}},
				{Kind: model.MessageKindLLM, Content: []model.ContentPart{model.NewContentText("x")}},
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

func testTextParts(text string) []model.ContentPart {
	return []model.ContentPart{model.NewContentText(text)}
}
