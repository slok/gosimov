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
			line := sessionToLine(test.session)
			got := lineToSession(line)

			assert.Equal(t, test.session.ID, got.ID)
			assert.Equal(t, test.session.CreatedAt.Unix(), got.CreatedAt.Unix())

			// Verify the line type is set.
			assert.Equal(t, lineTypeSession, line.Type)
		})
	}
}

func TestSessionLineJSON(t *testing.T) {
	session := model.Session{
		ID:        "01KHXABC",
		CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
	}

	line := sessionToLine(session)
	data, err := json.Marshal(line)
	require.NoError(t, err)

	// Verify it contains expected fields.
	var raw map[string]any
	require.NoError(t, json.Unmarshal(data, &raw))
	assert.Equal(t, "session", raw["type"])
	assert.Equal(t, "01KHXABC", raw["id"])
	assert.NotEmpty(t, raw["created_at"])

	// Round-trip through JSON.
	var parsed sessionLine
	require.NoError(t, json.Unmarshal(data, &parsed))
	got := lineToSession(parsed)
	assert.Equal(t, session.ID, got.ID)
}

func TestMessageRoundTrip(t *testing.T) {
	tests := map[string]struct {
		msg model.Message
	}{
		"User message with text content.": {
			msg: model.Message{
				ID:        "m1",
				Kind:      model.MessageKindUser,
				Content:   []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello world"}},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 1, 0, time.UTC),
			},
		},

		"LLM message with text and metadata.": {
			msg: model.Message{
				ID:      "m2",
				Kind:    model.MessageKindLLM,
				Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "I can help"}},
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
				Content:    []model.ContentPart{{Type: model.ContentPartTypeText, Text: "file contents"}},
				ToolCallID: "tc1",
				CreatedAt:  time.Date(2026, 2, 20, 16, 0, 4, 0, time.UTC),
			},
		},

		"Error tool result message.": {
			msg: model.Message{
				ID:         "m5",
				Kind:       model.MessageKindToolResult,
				Content:    []model.ContentPart{{Type: model.ContentPartTypeText, Text: "file not found"}},
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
					{Type: model.ContentPartTypeText, Text: "look at this"},
					{Type: model.ContentPartTypeImage, Image: &model.ImageData{
						Data:     []byte{0x89, 0x50, 0x4E, 0x47}, // PNG header bytes.
						MimeType: "image/png",
					}},
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 6, 0, time.UTC),
			},
		},

		"LLM message with full usage metadata.": {
			msg: model.Message{
				ID:      "m7",
				Kind:    model.MessageKindLLM,
				Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "done"}},
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
						CostUSD:          0.005,
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
				Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "Summary of the conversation so far"}},
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
				Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "Minimal compaction"}},
				Compaction: &model.CompactionData{
					FirstKeptID: "m1",
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 10, 0, time.UTC),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// Convert to line and back.
			line := messageToLine(test.msg)
			got := lineToMessage(line)

			// Verify type is set.
			assert.Equal(t, lineTypeMessage, line.Type)

			// Core fields.
			assert.Equal(t, test.msg.ID, got.ID)
			assert.Equal(t, test.msg.Kind, got.Kind)
			assert.Equal(t, test.msg.ToolCallID, got.ToolCallID)
			assert.Equal(t, test.msg.IsError, got.IsError)
			assert.Equal(t, test.msg.CreatedAt.Unix(), got.CreatedAt.Unix())

			// Content.
			require.Len(t, got.Content, len(test.msg.Content))
			for i, cp := range got.Content {
				assert.Equal(t, test.msg.Content[i].Type, cp.Type)
				assert.Equal(t, test.msg.Content[i].Text, cp.Text)
				if test.msg.Content[i].Image != nil {
					require.NotNil(t, cp.Image)
					assert.Equal(t, test.msg.Content[i].Image.Data, cp.Image.Data)
					assert.Equal(t, test.msg.Content[i].Image.MimeType, cp.Image.MimeType)
				}
			}

			// Tool call requests.
			require.Len(t, got.ToolCallRequests, len(test.msg.ToolCallRequests))
			for i, tc := range got.ToolCallRequests {
				assert.Equal(t, test.msg.ToolCallRequests[i].ID, tc.ID)
				assert.Equal(t, test.msg.ToolCallRequests[i].ToolID, tc.ToolID)
				assert.JSONEq(t, string(test.msg.ToolCallRequests[i].Arguments), string(tc.Arguments))
			}

			// Metadata.
			if test.msg.Metadata != nil {
				require.NotNil(t, got.Metadata)
				assert.Equal(t, test.msg.Metadata.StopReason, got.Metadata.StopReason)
				assert.Equal(t, test.msg.Metadata.Model, got.Metadata.Model)
				assert.Equal(t, test.msg.Metadata.Provider, got.Metadata.Provider)

				if test.msg.Metadata.Usage != nil {
					require.NotNil(t, got.Metadata.Usage)
					assert.Equal(t, *test.msg.Metadata.Usage, *got.Metadata.Usage)
				}
			} else {
				assert.Nil(t, got.Metadata)
			}

			// Compaction data.
			if test.msg.Compaction != nil {
				require.NotNil(t, got.Compaction)
				assert.Equal(t, test.msg.Compaction.FirstKeptID, got.Compaction.FirstKeptID)
				assert.Equal(t, test.msg.Compaction.TokensBefore, got.Compaction.TokensBefore)
			} else {
				assert.Nil(t, got.Compaction)
			}
		})
	}
}

func TestMessageLineJSONRoundTrip(t *testing.T) {
	msg := model.Message{
		ID:      "m1",
		Kind:    model.MessageKindLLM,
		Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}},
		ToolCallRequests: []model.ToolCallRequest{
			{ID: "tc1", ToolID: "read", Arguments: json.RawMessage(`{"path":"foo"}`)},
		},
		Metadata: &model.MessageMetadata{
			StopReason: model.StopReasonToolUse,
			Model:      "test",
			Usage:      &model.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
		},
		CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
	}

	// Marshal to JSON.
	line := messageToLine(msg)
	data, err := json.Marshal(line)
	require.NoError(t, err)

	// Verify JSON structure.
	var raw map[string]any
	require.NoError(t, json.Unmarshal(data, &raw))
	assert.Equal(t, "message", raw["type"])
	assert.Equal(t, "m1", raw["id"])
	assert.Equal(t, "llm", raw["kind"])

	// Unmarshal back and convert.
	var parsed messageLine
	require.NoError(t, json.Unmarshal(data, &parsed))
	got := lineToMessage(parsed)

	assert.Equal(t, msg.ID, got.ID)
	assert.Equal(t, msg.Kind, got.Kind)
	require.Len(t, got.ToolCallRequests, 1)
	assert.Equal(t, "tc1", got.ToolCallRequests[0].ID)
}

func TestCompactionMessageLineJSONRoundTrip(t *testing.T) {
	msg := model.Message{
		ID:      "c1",
		Kind:    model.MessageKindCompaction,
		Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "Summary of conversation"}},
		Compaction: &model.CompactionData{
			FirstKeptID:  "m5",
			TokensBefore: 3500,
		},
		CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
	}

	// Marshal to JSON.
	line := messageToLine(msg)
	data, err := json.Marshal(line)
	require.NoError(t, err)

	// Verify JSON structure.
	var raw map[string]any
	require.NoError(t, json.Unmarshal(data, &raw))
	assert.Equal(t, "message", raw["type"])
	assert.Equal(t, "c1", raw["id"])
	assert.Equal(t, "compaction", raw["kind"])

	// Verify compaction field in JSON.
	compaction, ok := raw["compaction"].(map[string]any)
	require.True(t, ok, "compaction field should be present in JSON")
	assert.Equal(t, "m5", compaction["first_kept_id"])
	assert.Equal(t, float64(3500), compaction["tokens_before"])

	// Unmarshal back and convert.
	var parsed messageLine
	require.NoError(t, json.Unmarshal(data, &parsed))
	got := lineToMessage(parsed)

	assert.Equal(t, msg.ID, got.ID)
	assert.Equal(t, msg.Kind, got.Kind)
	require.NotNil(t, got.Compaction)
	assert.Equal(t, "m5", got.Compaction.FirstKeptID)
	assert.Equal(t, 3500, got.Compaction.TokensBefore)
	require.Len(t, got.Content, 1)
	assert.Equal(t, "Summary of conversation", got.Content[0].Text)
}
