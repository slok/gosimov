package memory_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/memory"
)

func TestCreateSession(t *testing.T) {
	tests := map[string]struct {
		setup    func(t *testing.T, ctx context.Context, r *memory.Repository)
		session  model.Session
		expErr   bool
		expErrIs error
	}{
		"Creating a session should store it.": {
			session: model.Session{ID: "s1", CreatedAt: time.Now()},
		},

		"Creating a duplicate session should return ErrAlreadyExists.": {
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
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
			r := memory.NewRepository()

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
		setup    func(t *testing.T, ctx context.Context, r *memory.Repository)
		id       string
		expErr   bool
		expErrIs error
	}{
		"Getting an existing session should return it.": {
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
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
			r := memory.NewRepository()

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
		setup   func(t *testing.T, ctx context.Context, r *memory.Repository)
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
				require := require.New(t)
				for _, id := range []string{"s1", "s2", "s3"} {
					require.NoError(r.CreateSession(ctx, model.Session{ID: id, CreatedAt: time.Now()}))
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
				require := require.New(t)
				for _, id := range []string{"s1", "s2", "s3"} {
					require.NoError(r.CreateSession(ctx, model.Session{ID: id, CreatedAt: time.Now()}))
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
				require := require.New(t)
				for _, id := range []string{"s1", "s2", "s3"} {
					require.NoError(r.CreateSession(ctx, model.Session{ID: id, CreatedAt: time.Now()}))
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

		"Cursor past the end should return empty.": {
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1", CreatedAt: time.Now()}))
			},
			opts: store.ListOpts{Cursor: "99"},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert := assert.New(t)
				assert.Empty(result.Items)
				assert.Empty(result.NextCursor)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			ctx := context.Background()
			r := memory.NewRepository()

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
		setup     func(t *testing.T, ctx context.Context, r *memory.Repository)
		sessionID string
		msgs      []model.Message
		expErr    bool
		expErrIs  error
	}{
		"Appending messages to an existing session should work.": {
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
			},
			sessionID: "s1",
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
		},

		"Appending to a non-existent session should return ErrNotFound.": {
			sessionID: "missing",
			msgs:      []model.Message{{ID: "m1"}},
			expErr:    true,
			expErrIs:  pkgerrors.ErrNotFound,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			ctx := context.Background()
			r := memory.NewRepository()

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
		setup     func(t *testing.T, ctx context.Context, r *memory.Repository)
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
					{ID: "m2", Kind: model.MessageKindLLM},
					{ID: "m3", Kind: model.MessageKindToolResult},
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
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

		"Paginating through all messages should work.": {
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
				require := require.New(t)
				require.NoError(r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1"}, {ID: "m2"}, {ID: "m3"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{Limit: 2},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				// First page: m1, m2.
				require.Len(result.Items, 2)
				assert.Equal("m1", result.Items[0].ID)
				assert.Equal("m2", result.Items[1].ID)
				assert.NotEmpty(result.NextCursor)
			},
		},

		"Second page should return remaining messages.": {
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
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
			setup: func(t *testing.T, ctx context.Context, r *memory.Repository) {
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
			r := memory.NewRepository()

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
