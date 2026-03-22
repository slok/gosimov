package model_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/model"
)

func TestCloneMessage(t *testing.T) {
	tests := map[string]struct {
		messageFn               func() model.Message
		mutateClonedFn          func(cloned *model.Message)
		expMessage              model.Message
		expNestedPointersCloned bool
	}{
		"Message with nested fields should be deep-cloned.": {
			messageFn: testCloneMessageFixture,
			mutateClonedFn: func(cloned *model.Message) {
				cloned.Content[0].Text = "changed"
				cloned.Content[1].Image.Data[0] = 9
				cloned.ToolCallRequests[0].Arguments[0] = 'X'
				cloned.Metadata.Usage.InputTokens = 999
				cloned.Compaction.FirstKeptID = "changed-id"
			},
			expMessage:              testCloneMessageFixture(),
			expNestedPointersCloned: true,
		},

		"Message without optional fields should preserve zero values.": {
			messageFn: func() model.Message {
				return model.Message{ID: "m-empty", Kind: model.MessageKindUser}
			},
			expMessage: model.Message{ID: "m-empty", Kind: model.MessageKindUser},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			message := test.messageFn()
			cloned := model.CloneMessage(message)

			assert.Equal(test.expMessage, cloned)

			if test.expNestedPointersCloned {
				require.Len(message.Content, 2)
				require.Len(cloned.Content, 2)

				require.NotNil(message.Content[1].Image)
				require.NotNil(cloned.Content[1].Image)
				assert.NotSame(message.Content[1].Image, cloned.Content[1].Image)

				require.NotNil(message.Metadata)
				require.NotNil(cloned.Metadata)
				assert.NotSame(message.Metadata, cloned.Metadata)

				require.NotNil(message.Metadata.Usage)
				require.NotNil(cloned.Metadata.Usage)
				assert.NotSame(message.Metadata.Usage, cloned.Metadata.Usage)

				require.NotNil(message.Compaction)
				require.NotNil(cloned.Compaction)
				assert.NotSame(message.Compaction, cloned.Compaction)
			}

			if test.mutateClonedFn != nil {
				test.mutateClonedFn(&cloned)
			}

			assert.Equal(test.expMessage, message)
		})
	}
}

func TestCloneMessages(t *testing.T) {
	tests := map[string]struct {
		messagesFn              func() []model.Message
		mutateClonedFn          func(cloned []model.Message)
		expMessages             []model.Message
		expNilMessages          bool
		expNestedPointersCloned bool
	}{
		"Nil input should return nil.": {
			messagesFn:     func() []model.Message { return nil },
			expNilMessages: true,
		},

		"Empty input should return nil.": {
			messagesFn:     func() []model.Message { return []model.Message{} },
			expNilMessages: true,
		},

		"Messages with nested fields should be deep-cloned.": {
			messagesFn: testCloneMessagesFixture,
			mutateClonedFn: func(cloned []model.Message) {
				cloned[0].Content[0].Text = "changed"
				cloned[0].Content[1].Image.Data[0] = 9
				cloned[0].ToolCallRequests[0].Arguments[0] = 'X'
				cloned[0].Metadata.Usage.OutputTokens = 99
				cloned[0].Compaction.FirstKeptID = "mutated"
			},
			expMessages:             testCloneMessagesFixture(),
			expNestedPointersCloned: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			messages := test.messagesFn()
			cloned := model.CloneMessages(messages)

			if test.expNilMessages {
				assert.Nil(cloned)
				return
			}

			assert.Equal(test.expMessages, cloned)

			if test.expNestedPointersCloned {
				require.NotNil(messages[0].Content[1].Image)
				require.NotNil(cloned[0].Content[1].Image)
				assert.NotSame(messages[0].Content[1].Image, cloned[0].Content[1].Image)

				require.NotNil(messages[0].Metadata)
				require.NotNil(cloned[0].Metadata)
				assert.NotSame(messages[0].Metadata, cloned[0].Metadata)

				require.NotNil(messages[0].Metadata.Usage)
				require.NotNil(cloned[0].Metadata.Usage)
				assert.NotSame(messages[0].Metadata.Usage, cloned[0].Metadata.Usage)

				require.NotNil(messages[0].Compaction)
				require.NotNil(cloned[0].Compaction)
				assert.NotSame(messages[0].Compaction, cloned[0].Compaction)
			}

			if test.mutateClonedFn != nil {
				test.mutateClonedFn(cloned)
			}

			assert.Equal(test.expMessages, messages)
		})
	}
}

func testCloneMessageFixture() model.Message {
	return model.Message{
		ID:   "m1",
		Kind: model.MessageKindLLM,
		Content: []model.ContentPart{
			model.NewContentText("hello"),
			model.NewContentImage([]byte{1, 2, 3}, "image/png"),
		},
		ToolCallRequests: []model.ToolCallRequest{{
			ID:        "tc1",
			ToolID:    "tool1",
			Arguments: json.RawMessage(`{"k":"v"}`),
		}},
		Metadata: &model.MessageMetadata{
			Usage: &model.Usage{InputTokens: 10, OutputTokens: 20},
		},
		Compaction: &model.CompactionData{FirstKeptID: "m2", TokensBefore: 100},
	}
}

func testCloneMessagesFixture() []model.Message {
	return []model.Message{
		testCloneMessageFixture(),
		{ID: "m2", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("follow up")}},
	}
}
