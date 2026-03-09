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
			assert := assert.New(t)
			require := require.New(t)

			_, err := jsonl.New(test.cfg)

			if test.expErr {
				assert.Error(err)
				return
			}

			require.NoError(err)

			// Verify directory exists.
			info, err := os.Stat(test.cfg.Dir)
			require.NoError(err)
			assert.True(info.IsDir())
		})
	}
}

func TestCreateSession(t *testing.T) {
	tests := map[string]struct {
		setup    func(t *testing.T, ctx context.Context, r *jsonl.Repository)
		session  model.Session
		expErr   bool
		expErrIs error
	}{
		"Creating a session should store it.": {
			session: model.Session{ID: "s1", CreatedAt: time.Now()},
		},

		"Creating a duplicate session should return ErrAlreadyExists.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1", CreatedAt: time.Now()}))
			},
			session:  model.Session{ID: "s1", CreatedAt: time.Now()},
			expErr:   true,
			expErrIs: pkgerrors.ErrAlreadyExists,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(t, ctx, r)
			}

			err := r.CreateSession(ctx, test.session)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				return
			}

			require.NoError(err)

			// Verify it was stored.
			got, err := r.GetSession(ctx, test.session.ID)
			require.NoError(err)
			assert.Equal(test.session.ID, got.ID)
			assert.Equal(test.session.CreatedAt.Unix(), got.CreatedAt.Unix())
		})
	}
}

func TestGetSession(t *testing.T) {
	tests := map[string]struct {
		setup    func(t *testing.T, ctx context.Context, r *jsonl.Repository)
		id       string
		expErr   bool
		expErrIs error
	}{
		"Getting an existing session should return it.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1", CreatedAt: time.Now()}))
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
			assert := assert.New(t)
			require := require.New(t)

			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(t, ctx, r)
			}

			got, err := r.GetSession(ctx, test.id)

			if test.expErr {
				assert.Error(err)
				assert.Nil(got)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				return
			}

			require.NoError(err)
			assert.Equal(test.id, got.ID)
		})
	}
}

func TestListSessions(t *testing.T) {
	tests := map[string]struct {
		setup   func(t *testing.T, ctx context.Context, r *jsonl.Repository)
		opts    store.ListOpts
		expResp func(t *testing.T, result *store.ListResult[model.Session])
	}{
		"No sessions should return empty list.": {
			opts: store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert := assert.New(t)
				assert.Empty(result.Items)
				assert.Empty(result.NextCursor)
			},
		},

		"All sessions should be returned newest first.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				// Create with increasing timestamps.
				for i, id := range []string{"s1", "s2", "s3"} {
					require.NoError(r.CreateSession(ctx, model.Session{
						ID:        id,
						CreatedAt: time.Date(2026, 2, 20, 16, 0, i, 0, time.UTC),
					}))
				}
			},
			opts: store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 3)
				assert.Equal("s3", result.Items[0].ID)
				assert.Equal("s2", result.Items[1].ID)
				assert.Equal("s1", result.Items[2].ID)
				assert.Empty(result.NextCursor)
			},
		},

		"Limit should cap the number of results.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				for i, id := range []string{"s1", "s2", "s3"} {
					require.NoError(r.CreateSession(ctx, model.Session{
						ID:        id,
						CreatedAt: time.Date(2026, 2, 20, 16, 0, i, 0, time.UTC),
					}))
				}
			},
			opts: store.ListOpts{Limit: 2},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 2)
				assert.Equal("s3", result.Items[0].ID)
				assert.Equal("s2", result.Items[1].ID)
				assert.NotEmpty(result.NextCursor)
			},
		},

		"Cursor should resume from the right position.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				for i, id := range []string{"s1", "s2", "s3"} {
					require.NoError(r.CreateSession(ctx, model.Session{
						ID:        id,
						CreatedAt: time.Date(2026, 2, 20, 16, 0, i, 0, time.UTC),
					}))
				}
			},
			opts: store.ListOpts{Cursor: "2", Limit: 10},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 1)
				assert.Equal("s1", result.Items[0].ID)
				assert.Empty(result.NextCursor)
			},
		},

		"Non-jsonl files in directory should be ignored.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{
					ID:        "s1",
					CreatedAt: time.Now(),
				}))
			},
			opts: store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 1)
				assert.Equal("s1", result.Items[0].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(t, ctx, r)
			}

			result, err := r.ListSessions(ctx, test.opts)
			require.NoError(err)
			test.expResp(t, result)
		})
	}
}

func TestStoreMessages(t *testing.T) {
	tests := map[string]struct {
		setup     func(t *testing.T, ctx context.Context, r *jsonl.Repository)
		sessionID string
		msgs      []model.Message
		expErr    bool
		expErrIs  error
	}{
		"Appending messages to an existing session should work.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
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
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
			},
			sessionID: "s1",
			msgs:      []model.Message{},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(t, ctx, r)
			}

			err := r.StoreMessages(ctx, test.sessionID, test.msgs)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				return
			}

			require.NoError(err)

			// Verify messages were stored.
			result, err := r.ListMessages(ctx, test.sessionID, store.ListOpts{})
			require.NoError(err)
			assert.Len(result.Items, len(test.msgs))
		})
	}
}

func TestListMessages(t *testing.T) {
	tests := map[string]struct {
		setup     func(t *testing.T, ctx context.Context, r *jsonl.Repository)
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
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert := assert.New(t)
				assert.Empty(result.Items)
				assert.Empty(result.NextCursor)
			},
		},

		"All messages should be returned in insertion order.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
					{ID: "m2", Kind: model.MessageKindLLM},
					{ID: "m3", Kind: model.MessageKindToolResult, ToolCallID: "tc1"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 3)
				assert.Equal("m1", result.Items[0].ID)
				assert.Equal("m2", result.Items[1].ID)
				assert.Equal("m3", result.Items[2].ID)
				assert.Empty(result.NextCursor)
			},
		},

		"Limit should cap the number of messages.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1"}, {ID: "m2"}, {ID: "m3"}, {ID: "m4"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{Limit: 2},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 2)
				assert.Equal("m1", result.Items[0].ID)
				assert.Equal("m2", result.Items[1].ID)
				assert.NotEmpty(result.NextCursor)
			},
		},

		"Cursor should resume from the right position.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1"}, {ID: "m2"}, {ID: "m3"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{Cursor: "2", Limit: 10},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 1)
				assert.Equal("m3", result.Items[0].ID)
				assert.Empty(result.NextCursor)
			},
		},

		"Multiple appends should accumulate in order.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{{ID: "m2"}, {ID: "m3"}}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 3)
				assert.Equal("m1", result.Items[0].ID)
				assert.Equal("m2", result.Items[1].ID)
				assert.Equal("m3", result.Items[2].ID)
			},
		},

		"Messages from different sessions should be isolated.": {
			setup: func(t *testing.T, ctx context.Context, r *jsonl.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s2"}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}, {ID: "m2"}}))
				require.NoError(r.StoreMessages(ctx, "s2", []model.Message{{ID: "m3"}}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				require.Len(result.Items, 2)
				assert.Equal("m1", result.Items[0].ID)
				assert.Equal("m2", result.Items[1].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			ctx := context.Background()
			r := newRepo(t)

			if test.setup != nil {
				test.setup(t, ctx, r)
			}

			result, err := r.ListMessages(ctx, test.sessionID, test.opts)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				return
			}

			require.NoError(err)
			test.expResp(t, result)
		})
	}
}

func TestFileContents(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Session file should be valid JSONL with session header as first line.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				ctx := context.Background()
				dir := t.TempDir()
				r, err := jsonl.New(jsonl.Config{Dir: dir})
				require.NoError(err)

				session := model.Session{
					ID:        "test-session",
					CreatedAt: time.Date(2026, 2, 20, 16, 0, 0, 0, time.UTC),
				}
				require.NoError(r.CreateSession(ctx, session))
				require.NoError(r.StoreMessages(ctx, "test-session", []model.Message{
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

				data, err := os.ReadFile(filepath.Join(dir, "test-session.jsonl"))
				require.NoError(err)

				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				require.Len(lines, 3, "expected 1 session header + 2 messages")

				var header map[string]any
				require.NoError(json.Unmarshal([]byte(lines[0]), &header))
				assert.Equal("session", header["type"])
				assert.Equal("test-session", header["id"])

				var msg1 map[string]any
				require.NoError(json.Unmarshal([]byte(lines[1]), &msg1))
				assert.Equal("message", msg1["type"])
				assert.Equal("m1", msg1["id"])
				assert.Equal("user", msg1["kind"])

				var msg2 map[string]any
				require.NoError(json.Unmarshal([]byte(lines[2]), &msg2))
				assert.Equal("message", msg2["type"])
				assert.Equal("m2", msg2["id"])
				assert.Equal("llm", msg2["kind"])

				metadata, ok := msg2["metadata"].(map[string]any)
				require.True(ok)
				assert.Equal("complete", metadata["stop_reason"])
				assert.Equal("test-model", metadata["model"])
			},
		},

		"Compaction message should be preserved in JSONL.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				ctx := context.Background()
				dir := t.TempDir()
				r, err := jsonl.New(jsonl.Config{Dir: dir})
				require.NoError(err)

				require.NoError(r.CreateSession(ctx, model.Session{ID: "compact-session"}))
				require.NoError(r.StoreMessages(ctx, "compact-session", []model.Message{
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

				result, err := r.ListMessages(ctx, "compact-session", store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 4)

				compMsg := result.Items[2]
				assert.Equal("c1", compMsg.ID)
				assert.Equal(model.MessageKindCompaction, compMsg.Kind)
				require.Len(compMsg.Content, 1)
				assert.Equal("Summary of earlier conversation", compMsg.Content[0].Text)
				require.NotNil(compMsg.Compaction)
				assert.Equal("m2", compMsg.Compaction.FirstKeptID)
				assert.Equal(5000, compMsg.Compaction.TokensBefore)

				data, err := os.ReadFile(filepath.Join(dir, "compact-session.jsonl"))
				require.NoError(err)
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				require.Len(lines, 5, "expected 1 session header + 4 messages")

				var compLine map[string]any
				require.NoError(json.Unmarshal([]byte(lines[3]), &compLine))
				assert.Equal("compaction", compLine["kind"])
				compData, ok := compLine["compaction"].(map[string]any)
				require.True(ok)
				assert.Equal("m2", compData["first_kept_id"])
				assert.Equal(float64(5000), compData["tokens_before"])
			},
		},

		"Message with image data should be preserved in JSONL.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				ctx := context.Background()
				dir := t.TempDir()
				r, err := jsonl.New(jsonl.Config{Dir: dir})
				require.NoError(err)

				require.NoError(r.CreateSession(ctx, model.Session{ID: "img-session"}))
				imgData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
				require.NoError(r.StoreMessages(ctx, "img-session", []model.Message{
					{
						ID:   "m1",
						Kind: model.MessageKindUser,
						Content: []model.ContentPart{
							{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: imgData, MimeType: "image/png"}},
						},
					},
				}))

				result, err := r.ListMessages(ctx, "img-session", store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 1)
				require.Len(result.Items[0].Content, 1)
				require.NotNil(result.Items[0].Content[0].Image)
				assert.Equal(imgData, result.Items[0].Content[0].Image.Data)
				assert.Equal("image/png", result.Items[0].Content[0].Image.MimeType)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

// --- Test helpers ---

// newRepo creates a new JSONL repository using t.TempDir.
func newRepo(t *testing.T) *jsonl.Repository {
	t.Helper()

	require := require.New(t)

	r, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(err)

	return r
}
