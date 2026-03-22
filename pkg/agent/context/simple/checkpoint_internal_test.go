package simple

import (
	"testing"

	"github.com/slok/gosimov/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLatestCheckpoint(t *testing.T) {
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

			idx, cp := latestCheckpoint(test.msgs)
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

func TestApplyLatestCheckpoint(t *testing.T) {
	tests := map[string]struct {
		msgs   []model.Message
		assert func(t *testing.T, got []model.Message)
	}{
		"No checkpoint should return all messages.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
			assert: func(t *testing.T, got []model.Message) {
				require.Len(t, got, 2)
				assert.Equal(t, "m1", got[0].ID)
				assert.Equal(t, "m2", got[1].ID)
			},
		},
		"Unknown first kept id should return all messages.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "missing"}},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
			assert: func(t *testing.T, got []model.Message) {
				require.Len(t, got, 3)
				assert.Equal(t, "m1", got[0].ID)
				assert.Equal(t, "c1", got[1].ID)
				assert.Equal(t, "m2", got[2].ID)
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
			assert: func(t *testing.T, got []model.Message) {
				require.Len(t, got, 2)
				assert.Equal(t, "c2", got[0].ID)
				assert.Equal(t, "m3", got[1].ID)
			},
		},
		"Checkpoint in kept range should not be duplicated.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "m2", Kind: model.MessageKindLLM},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
				{ID: "m3", Kind: model.MessageKindUser},
			},
			assert: func(t *testing.T, got []model.Message) {
				require.Len(t, got, 3)
				assert.Equal(t, "c1", got[0].ID)
				assert.Equal(t, "m2", got[1].ID)
				assert.Equal(t, "m3", got[2].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := applyLatestCheckpoint(test.msgs)
			test.assert(t, got)
		})
	}
}

func TestLatestSummaryText(t *testing.T) {
	tests := map[string]struct {
		msgs []model.Message
		exp  string
	}{
		"No checkpoint should return empty.": {
			msgs: []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
			exp:  "",
		},
		"Latest checkpoint should return first text.": {
			msgs: []model.Message{
				{ID: "c1", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("old")}, Compaction: &model.CompactionData{FirstKeptID: "m1"}},
				{ID: "c2", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("new")}, Compaction: &model.CompactionData{FirstKeptID: "m2"}},
			},
			exp: "new",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.exp, latestSummaryText(test.msgs))
		})
	}
}

func TestCreateCheckpoint(t *testing.T) {
	got := createCheckpoint("summary", "m3", 42)

	require.NotEmpty(t, got.ID)
	assert.Equal(t, model.MessageKindCompaction, got.Kind)
	require.Len(t, got.Content, 1)
	assert.Equal(t, "summary", got.Content[0].Text)
	require.NotNil(t, got.Compaction)
	assert.Equal(t, "m3", got.Compaction.FirstKeptID)
	assert.Equal(t, 42, got.Compaction.TokensBefore)
	assert.False(t, got.CreatedAt.IsZero())
}

func TestPrependCheckpoint(t *testing.T) {
	checkpoint := model.Message{ID: "c1", Kind: model.MessageKindCompaction}
	messages := []model.Message{{ID: "m1"}, {ID: "m2"}}

	got := prependCheckpoint(messages, checkpoint)

	require.Len(t, got, 3)
	assert.Equal(t, "c1", got[0].ID)
	assert.Equal(t, "m1", got[1].ID)
	assert.Equal(t, "m2", got[2].ID)
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
