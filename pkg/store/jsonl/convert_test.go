package jsonl

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/model"
)

func TestSessionRoundTrip(t *testing.T) {
	tests := map[string]struct {
		session model.Session
	}{
		"Basic session should round-trip correctly.": {
			session: model.Session{
				ID:        "01KHXABC",
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
			},
		},

		"Session with zero time should round-trip.": {
			session: model.Session{
				ID: "01KHXDEF",
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			line := sessionToLine(test.session)
			got := lineToSession(line)

			assert.Equal(test.session.ID, got.ID)
			assert.Equal(test.session.CreatedAt.Unix(), got.CreatedAt.Unix())

			// Verify the line type is set.
			assert.Equal(lineTypeSession, line.Type)
		})
	}
}

func TestSessionLineJSON(t *testing.T) {
	tests := map[string]struct {
		session model.Session
	}{
		"Session line should marshal to JSON with correct type and round-trip.": {
			session: model.Session{
				ID:        "01KHXABC",
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			line := sessionToLine(test.session)
			data, err := json.Marshal(line)
			require.NoError(err)

			var raw map[string]any
			require.NoError(json.Unmarshal(data, &raw))
			assert.Equal("session", raw["type"])
			assert.Equal(test.session.ID, raw["id"])
			assert.NotEmpty(raw["created_at"])

			var parsed sessionLine
			require.NoError(json.Unmarshal(data, &parsed))
			got := lineToSession(parsed)
			assert.Equal(test.session.ID, got.ID)
		})
	}
}

func TestMessageRoundTrip(t *testing.T) {
	tests := map[string]struct {
		msg model.Message
	}{
		"User message with text content.": {
			msg: model.Message{
				ID:        "m1",
				Kind:      model.MessageKindUser,
				Content:   []model.ContentPart{model.NewContentText("hello world")},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 1, 0, time.UTC),
			},
		},

		"LLM message with text and metadata.": {
			msg: model.Message{
				ID:      "m2",
				Kind:    model.MessageKindLLM,
				Content: []model.ContentPart{model.NewContentText("I can help")},
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonComplete,
					Model:      "kimi-k2.5-free",
					Provider:   "zen",
					Usage: &model.Usage{
						InputTokens:  100,
						OutputTokens: 50,
						TotalTokens:  150,
					},
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 2, 0, time.UTC),
			},
		},

		"LLM message with tool calls.": {
			msg: model.Message{
				ID:   "m3",
				Kind: model.MessageKindLLM,
				ToolCallRequests: []model.ToolCallRequest{
					{
						ID:        "tc1",
						ToolID:    "read",
						Arguments: json.RawMessage(`{"path":"main.go"}`),
					},
					{
						ID:        "tc2",
						ToolID:    "shell",
						Arguments: json.RawMessage(`{"command":"go build"}`),
					},
				},
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonToolUse,
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 3, 0, time.UTC),
			},
		},

		"Tool result message.": {
			msg: model.Message{
				ID:         "m4",
				Kind:       model.MessageKindToolResult,
				Content:    []model.ContentPart{model.NewContentText("file contents")},
				ToolCallID: "tc1",
				CreatedAt:  time.Date(2026, 2, 20, 16, 0, 4, 0, time.UTC),
			},
		},

		"Error tool result message.": {
			msg: model.Message{
				ID:         "m5",
				Kind:       model.MessageKindToolResult,
				Content:    []model.ContentPart{model.NewContentText("file not found")},
				ToolCallID: "tc1",
				IsError:    true,
				CreatedAt:  time.Date(2026, 2, 20, 16, 0, 5, 0, time.UTC),
			},
		},

		"Message with image content.": {
			msg: model.Message{
				ID:   "m6",
				Kind: model.MessageKindUser,
				Content: []model.ContentPart{
					model.NewContentText("look at this"),
					model.NewContentImage([]byte{0x89, 0x50, 0x4E, 0x47}, "image/png"), // PNG header bytes.
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 6, 0, time.UTC),
			},
		},

		"LLM message with full usage metadata.": {
			msg: model.Message{
				ID:      "m7",
				Kind:    model.MessageKindLLM,
				Content: []model.ContentPart{model.NewContentText("done")},
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonComplete,
					Model:      "gpt-4o",
					Provider:   "openai",
					Usage: &model.Usage{
						InputTokens:      200,
						OutputTokens:     100,
						CacheReadTokens:  50,
						CacheWriteTokens: 30,
						TotalTokens:      380,
						ReasoningTokens:  20,
					},
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 7, 0, time.UTC),
			},
		},

		"Empty message.": {
			msg: model.Message{
				ID:        "m8",
				Kind:      model.MessageKindUser,
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 8, 0, time.UTC),
			},
		},

		"Compaction message with summary and compaction data.": {
			msg: model.Message{
				ID:      "m9",
				Kind:    model.MessageKindCompaction,
				Content: []model.ContentPart{model.NewContentText("Summary of the conversation so far")},
				Compaction: &model.CompactionData{
					FirstKeptID:  "m5",
					TokensBefore: 4200,
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 9, 0, time.UTC),
			},
		},

		"Compaction message with zero tokens before.": {
			msg: model.Message{
				ID:      "m10",
				Kind:    model.MessageKindCompaction,
				Content: []model.ContentPart{model.NewContentText("Minimal compaction")},
				Compaction: &model.CompactionData{
					FirstKeptID: "m1",
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 10, 0, time.UTC),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			// Convert to line and back.
			line := messageToLine(test.msg)
			got := lineToMessage(line)

			// Verify type is set.
			assert.Equal(lineTypeMessage, line.Type)

			// Core fields.
			assert.Equal(test.msg.ID, got.ID)
			assert.Equal(test.msg.Kind, got.Kind)
			assert.Equal(test.msg.ToolCallID, got.ToolCallID)
			assert.Equal(test.msg.IsError, got.IsError)
			assert.Equal(test.msg.CreatedAt.Unix(), got.CreatedAt.Unix())

			// Content.
			require.Len(got.Content, len(test.msg.Content))
			for i, cp := range got.Content {
				assert.Equal(test.msg.Content[i].Type, cp.Type)
				assert.Equal(test.msg.Content[i].Text, cp.Text)
				if test.msg.Content[i].Image != nil {
					require.NotNil(cp.Image)
					assert.Equal(test.msg.Content[i].Image.Data, cp.Image.Data)
					assert.Equal(test.msg.Content[i].Image.MimeType, cp.Image.MimeType)
				}
			}

			// Tool call requests.
			require.Len(got.ToolCallRequests, len(test.msg.ToolCallRequests))
			for i, tc := range got.ToolCallRequests {
				assert.Equal(test.msg.ToolCallRequests[i].ID, tc.ID)
				assert.Equal(test.msg.ToolCallRequests[i].ToolID, tc.ToolID)
				assert.JSONEq(string(test.msg.ToolCallRequests[i].Arguments), string(tc.Arguments))
			}

			// Metadata.
			if test.msg.Metadata != nil {
				require.NotNil(got.Metadata)
				assert.Equal(test.msg.Metadata.StopReason, got.Metadata.StopReason)
				assert.Equal(test.msg.Metadata.Model, got.Metadata.Model)
				assert.Equal(test.msg.Metadata.Provider, got.Metadata.Provider)

				if test.msg.Metadata.Usage != nil {
					require.NotNil(got.Metadata.Usage)
					assert.Equal(*test.msg.Metadata.Usage, *got.Metadata.Usage)
				}
			} else {
				assert.Nil(got.Metadata)
			}

			// Compaction data.
			if test.msg.Compaction != nil {
				require.NotNil(got.Compaction)
				assert.Equal(test.msg.Compaction.FirstKeptID, got.Compaction.FirstKeptID)
				assert.Equal(test.msg.Compaction.TokensBefore, got.Compaction.TokensBefore)
			} else {
				assert.Nil(got.Compaction)
			}
		})
	}
}

func TestMessageLineJSONRoundTrip(t *testing.T) {
	tests := map[string]struct {
		msg        model.Message
		expType    string
		expKind    string
		assertJSON func(t *testing.T, raw map[string]any)
		assertMsg  func(t *testing.T, got model.Message)
	}{
		"LLM message with tool calls should round-trip through JSON.": {
			msg: model.Message{
				ID:      "m1",
				Kind:    model.MessageKindLLM,
				Content: []model.ContentPart{model.NewContentText("hello")},
				ToolCallRequests: []model.ToolCallRequest{
					{ID: "tc1", ToolID: "read", Arguments: json.RawMessage(`{"path":"foo"}`)},
				},
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonToolUse,
					Model:      "test",
					Usage:      &model.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
			},
			expType: "message",
			expKind: "llm",
			assertMsg: func(t *testing.T, got model.Message) {
				t.Helper()
				require := require.New(t)
				assert := assert.New(t)
				require.Len(got.ToolCallRequests, 1)
				assert.Equal("tc1", got.ToolCallRequests[0].ID)
			},
		},

		"Compaction message should preserve compaction data through JSON.": {
			msg: model.Message{
				ID:      "c1",
				Kind:    model.MessageKindCompaction,
				Content: []model.ContentPart{model.NewContentText("Summary of conversation")},
				Compaction: &model.CompactionData{
					FirstKeptID:  "m5",
					TokensBefore: 3500,
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
			},
			expType: "message",
			expKind: "compaction",
			assertJSON: func(t *testing.T, raw map[string]any) {
				t.Helper()
				require := require.New(t)
				assert := assert.New(t)
				compaction, ok := raw["compaction"].(map[string]any)
				require.True(ok, "compaction field should be present in JSON")
				assert.Equal("m5", compaction["first_kept_id"])
				assert.Equal(float64(3500), compaction["tokens_before"])
			},
			assertMsg: func(t *testing.T, got model.Message) {
				t.Helper()
				require := require.New(t)
				assert := assert.New(t)
				require.NotNil(got.Compaction)
				assert.Equal("m5", got.Compaction.FirstKeptID)
				assert.Equal(3500, got.Compaction.TokensBefore)
				require.Len(got.Content, 1)
				assert.Equal("Summary of conversation", got.Content[0].Text)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			line := messageToLine(test.msg)
			data, err := json.Marshal(line)
			require.NoError(err)

			var raw map[string]any
			require.NoError(json.Unmarshal(data, &raw))
			assert.Equal(test.expType, raw["type"])
			assert.Equal(test.msg.ID, raw["id"])
			assert.Equal(test.expKind, raw["kind"])

			if test.assertJSON != nil {
				test.assertJSON(t, raw)
			}

			var parsed messageLine
			require.NoError(json.Unmarshal(data, &parsed))
			got := lineToMessage(parsed)

			assert.Equal(test.msg.ID, got.ID)
			assert.Equal(test.msg.Kind, got.Kind)

			if test.assertMsg != nil {
				test.assertMsg(t, got)
			}
		})
	}
}
