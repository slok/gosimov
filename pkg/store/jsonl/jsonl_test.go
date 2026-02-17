package jsonl_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/jsonl"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		cfg    jsonl.Config
		expErr bool
	}{
		"Valid config should succeed.": {
			cfg: jsonl.Config{Dir: t.TempDir()},
		},

		"Empty dir should fail.": {
			cfg:    jsonl.Config{},
			expErr: true,
		},

		"Non-existent directory should be created.": {
			cfg: jsonl.Config{Dir: filepath.Join(t.TempDir(), "sub", "dir")},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := jsonl.New(test.cfg)

			if test.expErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)

			// Verify directory exists.
			info, err := os.Stat(test.cfg.Dir)
			require.NoError(t, err)
			assert.True(t, info.IsDir())
		})
	}
}

func TestCreateSession(t *testing.T) {
	tests := map[string]struct {
		setup    func(ctx context.Context, r *jsonl.Repository)
		session  model.Session
		expErr   bool
		expErrIs error
	}{
		"Creating a session should store it.": {
			session: model.Session{ID: "s1", CreatedAt: time.Now()},
		},

		"Creating a duplicate session should return ErrAlreadyExists.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1", CreatedAt: time.Now()}))
			},
			session:  model.Session{ID: "s1", CreatedAt: time.Now()},
			expErr:   true,
			expErrIs: pkgerrors.ErrAlreadyExists,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(ctx, r)
			}

			err := r.CreateSession(ctx, test.session)

			if test.expErr {
				assert.Error(t, err)
				if test.expErrIs != nil {
					assert.ErrorIs(t, err, test.expErrIs)
				}
				return
			}

			require.NoError(t, err)

			// Verify it was stored.
			got, err := r.GetSession(ctx, test.session.ID)
			require.NoError(t, err)
			assert.Equal(t, test.session.ID, got.ID)
			assert.Equal(t, test.session.CreatedAt.Unix(), got.CreatedAt.Unix())
		})
	}
}

func TestGetSession(t *testing.T) {
	tests := map[string]struct {
		setup    func(ctx context.Context, r *jsonl.Repository)
		id       string
		expErr   bool
		expErrIs error
	}{
		"Getting an existing session should return it.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1", CreatedAt: time.Now()}))
			},
			id: "s1",
		},

		"Getting a non-existent session should return ErrNotFound.": {
			id:       "missing",
			expErr:   true,
			expErrIs: pkgerrors.ErrNotFound,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(ctx, r)
			}

			got, err := r.GetSession(ctx, test.id)

			if test.expErr {
				assert.Error(t, err)
				assert.Nil(t, got)
				if test.expErrIs != nil {
					assert.ErrorIs(t, err, test.expErrIs)
				}
				return
			}

			require.NoError(t, err)
			assert.Equal(t, test.id, got.ID)
		})
	}
}

func TestListSessions(t *testing.T) {
	tests := map[string]struct {
		setup   func(ctx context.Context, r *jsonl.Repository)
		opts    store.ListOpts
		expResp func(t *testing.T, result *store.ListResult[model.Session])
	}{
		"No sessions should return empty list.": {
			opts: store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert.Empty(t, result.Items)
				assert.Empty(t, result.NextCursor)
			},
		},

		"All sessions should be returned newest first.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				// Create with increasing timestamps.
				for i, id := range []string{"s1", "s2", "s3"} {
					require.NoError(t, r.CreateSession(ctx, model.Session{
						ID:        id,
						CreatedAt: time.Date(2026, 2, 20, 16, 0, i, 0, time.UTC),
					}))
				}
			},
			opts: store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				require.Len(t, result.Items, 3)
				assert.Equal(t, "s3", result.Items[0].ID)
				assert.Equal(t, "s2", result.Items[1].ID)
				assert.Equal(t, "s1", result.Items[2].ID)
				assert.Empty(t, result.NextCursor)
			},
		},

		"Limit should cap the number of results.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				for i, id := range []string{"s1", "s2", "s3"} {
					require.NoError(t, r.CreateSession(ctx, model.Session{
						ID:        id,
						CreatedAt: time.Date(2026, 2, 20, 16, 0, i, 0, time.UTC),
					}))
				}
			},
			opts: store.ListOpts{Limit: 2},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				require.Len(t, result.Items, 2)
				assert.Equal(t, "s3", result.Items[0].ID)
				assert.Equal(t, "s2", result.Items[1].ID)
				assert.NotEmpty(t, result.NextCursor)
			},
		},

		"Cursor should resume from the right position.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				for i, id := range []string{"s1", "s2", "s3"} {
					require.NoError(t, r.CreateSession(ctx, model.Session{
						ID:        id,
						CreatedAt: time.Date(2026, 2, 20, 16, 0, i, 0, time.UTC),
					}))
				}
			},
			opts: store.ListOpts{Cursor: "2", Limit: 10},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				require.Len(t, result.Items, 1)
				assert.Equal(t, "s1", result.Items[0].ID)
				assert.Empty(t, result.NextCursor)
			},
		},

		"Non-jsonl files in directory should be ignored.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{
					ID:        "s1",
					CreatedAt: time.Now(),
				}))
			},
			opts: store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				require.Len(t, result.Items, 1)
				assert.Equal(t, "s1", result.Items[0].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(ctx, r)
			}

			result, err := r.ListSessions(ctx, test.opts)
			require.NoError(t, err)
			test.expResp(t, result)
		})
	}
}

func TestStoreMessages(t *testing.T) {
	tests := map[string]struct {
		setup     func(ctx context.Context, r *jsonl.Repository)
		sessionID string
		msgs      []model.Message
		expErr    bool
		expErrIs  error
	}{
		"Appending messages to an existing session should work.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
			},
			sessionID: "s1",
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
				{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
			},
		},

		"Appending to a non-existent session should return ErrNotFound.": {
			sessionID: "missing",
			msgs:      []model.Message{{ID: "m1"}},
			expErr:    true,
			expErrIs:  pkgerrors.ErrNotFound,
		},

		"Appending empty messages should be a no-op.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
			},
			sessionID: "s1",
			msgs:      []model.Message{},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(ctx, r)
			}

			err := r.StoreMessages(ctx, test.sessionID, test.msgs)

			if test.expErr {
				assert.Error(t, err)
				if test.expErrIs != nil {
					assert.ErrorIs(t, err, test.expErrIs)
				}
				return
			}

			require.NoError(t, err)

			// Verify messages were stored.
			result, err := r.ListMessages(ctx, test.sessionID, store.ListOpts{})
			require.NoError(t, err)
			assert.Len(t, result.Items, len(test.msgs))
		})
	}
}

func TestListMessages(t *testing.T) {
	tests := map[string]struct {
		setup     func(ctx context.Context, r *jsonl.Repository)
		sessionID string
		opts      store.ListOpts
		expResp   func(t *testing.T, result *store.ListResult[model.Message])
		expErr    bool
		expErrIs  error
	}{
		"Listing messages for a non-existent session should return ErrNotFound.": {
			sessionID: "missing",
			opts:      store.ListOpts{},
			expErr:    true,
			expErrIs:  pkgerrors.ErrNotFound,
		},

		"Empty session should return empty list.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert.Empty(t, result.Items)
				assert.Empty(t, result.NextCursor)
			},
		},

		"All messages should be returned in insertion order.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
					{ID: "m2", Kind: model.MessageKindLLM},
					{ID: "m3", Kind: model.MessageKindToolResult, ToolCallID: "tc1"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				require.Len(t, result.Items, 3)
				assert.Equal(t, "m1", result.Items[0].ID)
				assert.Equal(t, "m2", result.Items[1].ID)
				assert.Equal(t, "m3", result.Items[2].ID)
				assert.Empty(t, result.NextCursor)
			},
		},

		"Limit should cap the number of messages.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1"}, {ID: "m2"}, {ID: "m3"}, {ID: "m4"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{Limit: 2},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				require.Len(t, result.Items, 2)
				assert.Equal(t, "m1", result.Items[0].ID)
				assert.Equal(t, "m2", result.Items[1].ID)
				assert.NotEmpty(t, result.NextCursor)
			},
		},

		"Cursor should resume from the right position.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1"}, {ID: "m2"}, {ID: "m3"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{Cursor: "2", Limit: 10},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				require.Len(t, result.Items, 1)
				assert.Equal(t, "m3", result.Items[0].ID)
				assert.Empty(t, result.NextCursor)
			},
		},

		"Multiple appends should accumulate in order.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{{ID: "m2"}, {ID: "m3"}}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				require.Len(t, result.Items, 3)
				assert.Equal(t, "m1", result.Items[0].ID)
				assert.Equal(t, "m2", result.Items[1].ID)
				assert.Equal(t, "m3", result.Items[2].ID)
			},
		},

		"Messages from different sessions should be isolated.": {
			setup: func(ctx context.Context, r *jsonl.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s2"}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}, {ID: "m2"}}))
				require.NoError(t, r.StoreMessages(ctx, "s2", []model.Message{{ID: "m3"}}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				require.Len(t, result.Items, 2)
				assert.Equal(t, "m1", result.Items[0].ID)
				assert.Equal(t, "m2", result.Items[1].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(ctx, r)
			}

			result, err := r.ListMessages(ctx, test.sessionID, test.opts)

			if test.expErr {
				assert.Error(t, err)
				if test.expErrIs != nil {
					assert.ErrorIs(t, err, test.expErrIs)
				}
				return
			}

			require.NoError(t, err)
			test.expResp(t, result)
		})
	}
}

func TestFileContents(t *testing.T) {
	t.Run("Session file should be valid JSONL with session header as first line.", func(t *testing.T) {
		ctx := context.Background()
		dir := t.TempDir()
		r, err := jsonl.New(jsonl.Config{Dir: dir})
		require.NoError(t, err)

		session := model.Session{
			ID:        "test-session",
			CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
		}
		require.NoError(t, r.CreateSession(ctx, session))
		require.NoError(t, r.StoreMessages(ctx, "test-session", []model.Message{
			{
				ID:        "m1",
				Kind:      model.MessageKindUser,
				Content:   []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 1, 0, time.UTC),
			},
			{
				ID:      "m2",
				Kind:    model.MessageKindLLM,
				Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi there"}},
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonComplete,
					Model:      "test-model",
					Provider:   "test-provider",
					Usage:      &model.Usage{InputTokens: 10, OutputTokens: 5},
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 2, 0, time.UTC),
			},
		}))

		// Read the file and verify structure.
		data, err := os.ReadFile(filepath.Join(dir, "test-session.jsonl"))
		require.NoError(t, err)

		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		require.Len(t, lines, 3, "expected 1 session header + 2 messages")

		// First line: session header.
		var header map[string]any
		require.NoError(t, json.Unmarshal([]byte(lines[0]), &header))
		assert.Equal(t, "session", header["type"])
		assert.Equal(t, "test-session", header["id"])

		// Second line: user message.
		var msg1 map[string]any
		require.NoError(t, json.Unmarshal([]byte(lines[1]), &msg1))
		assert.Equal(t, "message", msg1["type"])
		assert.Equal(t, "m1", msg1["id"])
		assert.Equal(t, "user", msg1["kind"])

		// Third line: LLM message.
		var msg2 map[string]any
		require.NoError(t, json.Unmarshal([]byte(lines[2]), &msg2))
		assert.Equal(t, "message", msg2["type"])
		assert.Equal(t, "m2", msg2["id"])
		assert.Equal(t, "llm", msg2["kind"])

		// Verify metadata is present.
		metadata, ok := msg2["metadata"].(map[string]any)
		require.True(t, ok)
		assert.Equal(t, "complete", metadata["stop_reason"])
		assert.Equal(t, "test-model", metadata["model"])
	})

	t.Run("Compaction message should be preserved in JSONL.", func(t *testing.T) {
		ctx := context.Background()
		dir := t.TempDir()
		r, err := jsonl.New(jsonl.Config{Dir: dir})
		require.NoError(t, err)

		require.NoError(t, r.CreateSession(ctx, model.Session{ID: "compact-session"}))
		require.NoError(t, r.StoreMessages(ctx, "compact-session", []model.Message{
			{
				ID:        "m1",
				Kind:      model.MessageKindUser,
				Content:   []model.ContentPart{{Type: model.ContentPartTypeText, Text: "old message"}},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
			},
			{
				ID:      "m2",
				Kind:    model.MessageKindLLM,
				Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "old response"}},
				Metadata: &model.MessageMetadata{
					StopReason: model.StopReasonComplete,
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 1, 0, time.UTC),
			},
			{
				ID:      "c1",
				Kind:    model.MessageKindCompaction,
				Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "Summary of earlier conversation"}},
				Compaction: &model.CompactionData{
					FirstKeptID:  "m2",
					TokensBefore: 5000,
				},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 2, 0, time.UTC),
			},
			{
				ID:        "m3",
				Kind:      model.MessageKindUser,
				Content:   []model.ContentPart{{Type: model.ContentPartTypeText, Text: "new message"}},
				CreatedAt: time.Date(2026, 2, 20, 16, 0, 3, 0, time.UTC),
			},
		}))

		// Read back and verify all messages including compaction.
		result, err := r.ListMessages(ctx, "compact-session", store.ListOpts{})
		require.NoError(t, err)
		require.Len(t, result.Items, 4)

		// Verify the compaction message (third in the list).
		compMsg := result.Items[2]
		assert.Equal(t, "c1", compMsg.ID)
		assert.Equal(t, model.MessageKindCompaction, compMsg.Kind)
		require.Len(t, compMsg.Content, 1)
		assert.Equal(t, "Summary of earlier conversation", compMsg.Content[0].Text)
		require.NotNil(t, compMsg.Compaction)
		assert.Equal(t, "m2", compMsg.Compaction.FirstKeptID)
		assert.Equal(t, 5000, compMsg.Compaction.TokensBefore)

		// Verify the JSONL file has the compaction data.
		data, err := os.ReadFile(filepath.Join(dir, "compact-session.jsonl"))
		require.NoError(t, err)
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		require.Len(t, lines, 5, "expected 1 session header + 4 messages")

		// Third message line (index 3) should be the compaction.
		var compLine map[string]any
		require.NoError(t, json.Unmarshal([]byte(lines[3]), &compLine))
		assert.Equal(t, "compaction", compLine["kind"])
		compData, ok := compLine["compaction"].(map[string]any)
		require.True(t, ok)
		assert.Equal(t, "m2", compData["first_kept_id"])
		assert.Equal(t, float64(5000), compData["tokens_before"])
	})

	t.Run("Message with image data should be preserved in JSONL.", func(t *testing.T) {
		ctx := context.Background()
		dir := t.TempDir()
		r, err := jsonl.New(jsonl.Config{Dir: dir})
		require.NoError(t, err)

		require.NoError(t, r.CreateSession(ctx, model.Session{ID: "img-session"}))
		imgData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A} // PNG header.
		require.NoError(t, r.StoreMessages(ctx, "img-session", []model.Message{
			{
				ID:   "m1",
				Kind: model.MessageKindUser,
				Content: []model.ContentPart{
					{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: imgData, MimeType: "image/png"}},
				},
			},
		}))

		// Read it back and verify image data is preserved.
		result, err := r.ListMessages(ctx, "img-session", store.ListOpts{})
		require.NoError(t, err)
		require.Len(t, result.Items, 1)
		require.Len(t, result.Items[0].Content, 1)
		require.NotNil(t, result.Items[0].Content[0].Image)
		assert.Equal(t, imgData, result.Items[0].Content[0].Image.Data)
		assert.Equal(t, "image/png", result.Items[0].Content[0].Image.MimeType)
	})
}

// --- Test helpers ---

// newRepo creates a new JSONL repository using t.TempDir.
func newRepo(t *testing.T) *jsonl.Repository {
	t.Helper()

	r, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	return r
}
