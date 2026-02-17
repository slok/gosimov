package subscriber_test

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
	"github.com/slok/gosimov/pkg/store/subscriber"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		config   subscriber.Config
		expErr   bool
		expErrIs error
	}{
		"Missing wrapped repository should fail.": {
			config:   subscriber.Config{},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Valid config should create repository.": {
			config: subscriber.Config{Repository: memory.NewRepository()},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			r, err := subscriber.New(test.config)

			if test.expErr {
				assert.Error(t, err)
				if test.expErrIs != nil {
					assert.ErrorIs(t, err, test.expErrIs)
				}
				assert.Nil(t, r)
				return
			}

			require.NoError(t, err)
			assert.NotNil(t, r)
		})
	}
}

func TestStoreMessages(t *testing.T) {
	tests := map[string]struct {
		setup     func(t *testing.T) (*subscriber.Repository, context.Context)
		sessionID string
		msgs      []model.Message
		expErr    bool
		expErrIs  error
		expEvent  func(t *testing.T, event subscriber.MessageStoredEvent)
	}{
		"Successful store should emit event.": {
			setup: func(t *testing.T) (*subscriber.Repository, context.Context) {
				t.Helper()
				ctx := context.Background()
				base := memory.NewRepository()
				require.NoError(t, base.CreateSession(ctx, model.Session{ID: "s1"}))

				repo, err := subscriber.New(subscriber.Config{Repository: base})
				require.NoError(t, err)

				return repo, ctx
			},
			sessionID: "s1",
			msgs:      []model.Message{{ID: "m1"}, {ID: "m2"}},
			expEvent: func(t *testing.T, event subscriber.MessageStoredEvent) {
				t.Helper()
				assert.Equal(t, "s1", event.SessionID)
				require.Len(t, event.Messages, 2)
				assert.Equal(t, "m1", event.Messages[0].ID)
				assert.Equal(t, "m2", event.Messages[1].ID)
				assert.False(t, event.Replay)
				assert.False(t, event.StoredAt.IsZero())
			},
		},

		"Store error should not emit event.": {
			setup: func(t *testing.T) (*subscriber.Repository, context.Context) {
				t.Helper()
				ctx := context.Background()
				base := memory.NewRepository()

				repo, err := subscriber.New(subscriber.Config{Repository: base})
				require.NoError(t, err)

				return repo, ctx
			},
			sessionID: "missing",
			msgs:      []model.Message{{ID: "m1"}},
			expErr:    true,
			expErrIs:  pkgerrors.ErrNotFound,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			repo, ctx := test.setup(t)

			subCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{})
			require.NoError(t, err)

			err = repo.StoreMessages(ctx, test.sessionID, test.msgs)

			if test.expErr {
				assert.Error(t, err)
				if test.expErrIs != nil {
					assert.ErrorIs(t, err, test.expErrIs)
				}

				assertNoEvent(t, events)
				return
			}

			require.NoError(t, err)
			event := readEvent(t, events)
			if test.expEvent != nil {
				test.expEvent(t, event)
			}
		})
	}
}

func TestSessionFilter(t *testing.T) {
	ctx := context.Background()
	base := memory.NewRepository()
	require.NoError(t, base.CreateSession(ctx, model.Session{ID: "s1"}))
	require.NoError(t, base.CreateSession(ctx, model.Session{ID: "s2"}))

	repo, err := subscriber.New(subscriber.Config{Repository: base})
	require.NoError(t, err)

	subCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{SessionID: "s1"})
	require.NoError(t, err)

	require.NoError(t, repo.StoreMessages(ctx, "s2", []model.Message{{ID: "m2"}}))
	assertNoEvent(t, events)

	require.NoError(t, repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}}))
	event := readEvent(t, events)
	assert.Equal(t, "s1", event.SessionID)
	require.Len(t, event.Messages, 1)
	assert.Equal(t, "m1", event.Messages[0].ID)
}

func TestSubscribeReplay(t *testing.T) {
	ctx := context.Background()
	base := memory.NewRepository()
	require.NoError(t, base.CreateSession(ctx, model.Session{ID: "s1"}))
	require.NoError(t, base.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}, {ID: "m2"}, {ID: "m3"}}))

	repo, err := subscriber.New(subscriber.Config{Repository: base, ReplayPageLimit: 2})
	require.NoError(t, err)

	subCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{SessionID: "s1", Replay: true})
	require.NoError(t, err)

	first := readEvent(t, events)
	assert.True(t, first.Replay)
	require.Len(t, first.Messages, 2)
	assert.Equal(t, "m1", first.Messages[0].ID)
	assert.Equal(t, "m2", first.Messages[1].ID)

	second := readEvent(t, events)
	assert.True(t, second.Replay)
	require.Len(t, second.Messages, 1)
	assert.Equal(t, "m3", second.Messages[0].ID)
}

func TestSubscribeReplayNeedsSession(t *testing.T) {
	base := memory.NewRepository()
	repo, err := subscriber.New(subscriber.Config{Repository: base})
	require.NoError(t, err)

	_, err = repo.Subscribe(context.Background(), subscriber.SubscribeOpts{Replay: true})
	assert.Error(t, err)
	assert.ErrorIs(t, err, pkgerrors.ErrNotValid)
}

func TestSubscribeCloseOnCancel(t *testing.T) {
	base := memory.NewRepository()
	repo, err := subscriber.New(subscriber.Config{Repository: base})
	require.NoError(t, err)

	subCtx, cancel := context.WithCancel(context.Background())
	events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{})
	require.NoError(t, err)

	cancel()

	select {
	case _, ok := <-events:
		assert.False(t, ok)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for closed channel")
	}
}

func TestDropOldestWhenBufferIsFull(t *testing.T) {
	ctx := context.Background()
	base := memory.NewRepository()
	require.NoError(t, base.CreateSession(ctx, model.Session{ID: "s1"}))

	repo, err := subscriber.New(subscriber.Config{Repository: base, BufferSize: 1})
	require.NoError(t, err)

	subCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{SessionID: "s1"})
	require.NoError(t, err)

	require.NoError(t, repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}}))
	require.NoError(t, repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m2"}}))
	require.NoError(t, repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m3"}}))

	event := readEvent(t, events)
	require.Len(t, event.Messages, 1)
	assert.Equal(t, "m3", event.Messages[0].ID)
}

func readEvent(t *testing.T, events <-chan subscriber.MessageStoredEvent) subscriber.MessageStoredEvent {
	t.Helper()

	select {
	case event := <-events:
		return event
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for event")
		return subscriber.MessageStoredEvent{}
	}
}

func assertNoEvent(t *testing.T, events <-chan subscriber.MessageStoredEvent) {
	t.Helper()

	select {
	case event := <-events:
		t.Fatalf("unexpected event: %+v", event)
	case <-time.After(100 * time.Millisecond):
	}
}

var _ store.MessageRepository = (*subscriber.Repository)(nil)
