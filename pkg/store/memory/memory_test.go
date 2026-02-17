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
		setup    func(ctx context.Context, r *memory.Repository)
		session  model.Session
		expErr   bool
		expErrIs error
	}{
		"Creating a session should store it.": {
			session: model.Session{ID: "s1", CreatedAt: time.Now()},
		},

		"Creating a duplicate session should return ErrAlreadyExists.": {
			setup: func(ctx context.Context, r *memory.Repository) {
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
			r := memory.NewRepository()

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
		setup    func(ctx context.Context, r *memory.Repository)
		id       string
		expErr   bool
		expErrIs error
	}{
		"Getting an existing session should return it.": {
			setup: func(ctx context.Context, r *memory.Repository) {
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
			r := memory.NewRepository()

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
		setup   func(ctx context.Context, r *memory.Repository)
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
			setup: func(ctx context.Context, r *memory.Repository) {
				for _, id := range []string{"s1", "s2", "s3"} {
					require.NoError(t, r.CreateSession(ctx, model.Session{ID: id, CreatedAt: time.Now()}))
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
			setup: func(ctx context.Context, r *memory.Repository) {
				for _, id := range []string{"s1", "s2", "s3"} {
					require.NoError(t, r.CreateSession(ctx, model.Session{ID: id, CreatedAt: time.Now()}))
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
			setup: func(ctx context.Context, r *memory.Repository) {
				for _, id := range []string{"s1", "s2", "s3"} {
					require.NoError(t, r.CreateSession(ctx, model.Session{ID: id, CreatedAt: time.Now()}))
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

		"Cursor past the end should return empty.": {
			setup: func(ctx context.Context, r *memory.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1", CreatedAt: time.Now()}))
			},
			opts: store.ListOpts{Cursor: "99"},
			expResp: func(t *testing.T, result *store.ListResult[model.Session]) {
				t.Helper()
				assert.Empty(t, result.Items)
				assert.Empty(t, result.NextCursor)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			r := memory.NewRepository()

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
		setup     func(ctx context.Context, r *memory.Repository)
		sessionID string
		msgs      []model.Message
		expErr    bool
		expErrIs  error
	}{
		"Appending messages to an existing session should work.": {
			setup: func(ctx context.Context, r *memory.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
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
			ctx := context.Background()
			r := memory.NewRepository()

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
		setup     func(ctx context.Context, r *memory.Repository)
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
			setup: func(ctx context.Context, r *memory.Repository) {
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
			setup: func(ctx context.Context, r *memory.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
					{ID: "m2", Kind: model.MessageKindLLM},
					{ID: "m3", Kind: model.MessageKindToolResult},
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
			setup: func(ctx context.Context, r *memory.Repository) {
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

		"Paginating through all messages should work.": {
			setup: func(ctx context.Context, r *memory.Repository) {
				require.NoError(t, r.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(t, r.StoreMessages(ctx, "s1", []model.Message{
					{ID: "m1"}, {ID: "m2"}, {ID: "m3"},
				}))
			},
			sessionID: "s1",
			opts:      store.ListOpts{Limit: 2},
			expResp: func(t *testing.T, result *store.ListResult[model.Message]) {
				t.Helper()

				// First page: m1, m2.
				require.Len(t, result.Items, 2)
				assert.Equal(t, "m1", result.Items[0].ID)
				assert.Equal(t, "m2", result.Items[1].ID)
				assert.NotEmpty(t, result.NextCursor)
			},
		},

		"Second page should return remaining messages.": {
			setup: func(ctx context.Context, r *memory.Repository) {
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
			setup: func(ctx context.Context, r *memory.Repository) {
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
			setup: func(ctx context.Context, r *memory.Repository) {
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
			r := memory.NewRepository()

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
