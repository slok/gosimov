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
			assert := assert.New(t)
			require := require.New(t)

			r, err := subscriber.New(test.config)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				assert.Nil(r)
				return
			}

			require.NoError(err)
			assert.NotNil(r)
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
				require := require.New(t)

				ctx := context.Background()
				base := memory.NewRepository()
				require.NoError(base.CreateSession(ctx, model.Session{ID: "s1"}))

				repo, err := subscriber.New(subscriber.Config{Repository: base})
				require.NoError(err)

				return repo, ctx
			},
			sessionID: "s1",
			msgs:      []model.Message{{ID: "m1"}, {ID: "m2"}},
			expEvent: func(t *testing.T, event subscriber.MessageStoredEvent) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				assert.Equal("s1", event.SessionID)
				require.Len(event.Messages, 2)
				assert.Equal("m1", event.Messages[0].ID)
				assert.Equal("m2", event.Messages[1].ID)
				assert.False(event.Replay)
				assert.False(event.StoredAt.IsZero())
			},
		},

		"Store error should not emit event.": {
			setup: func(t *testing.T) (*subscriber.Repository, context.Context) {
				t.Helper()
				require := require.New(t)

				ctx := context.Background()
				base := memory.NewRepository()

				repo, err := subscriber.New(subscriber.Config{Repository: base})
				require.NoError(err)

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
			assert := assert.New(t)
			require := require.New(t)

			repo, ctx := test.setup(t)

			subCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{})
			require.NoError(err)

			err = repo.StoreMessages(ctx, test.sessionID, test.msgs)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}

				assertNoEvent(t, events)
				return
			}

			require.NoError(err)
			event := readEvent(t, events)
			if test.expEvent != nil {
				test.expEvent(t, event)
			}
		})
	}
}

func TestSubscribeBehavior(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Session filter should only deliver events for the subscribed session.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				ctx := context.Background()
				base := memory.NewRepository()
				require.NoError(base.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(base.CreateSession(ctx, model.Session{ID: "s2"}))

				repo, err := subscriber.New(subscriber.Config{Repository: base})
				require.NoError(err)

				subCtx, cancel := context.WithCancel(context.Background())
				defer cancel()

				events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{SessionID: "s1"})
				require.NoError(err)

				require.NoError(repo.StoreMessages(ctx, "s2", []model.Message{{ID: "m2"}}))
				assertNoEvent(t, events)

				require.NoError(repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}}))
				event := readEvent(t, events)
				assert.Equal("s1", event.SessionID)
				require.Len(event.Messages, 1)
				assert.Equal("m1", event.Messages[0].ID)
			},
		},

		"Replay should deliver historical messages in pages then switch to live.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				ctx := context.Background()
				base := memory.NewRepository()
				require.NoError(base.CreateSession(ctx, model.Session{ID: "s1"}))
				require.NoError(base.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}, {ID: "m2"}, {ID: "m3"}}))

				repo, err := subscriber.New(subscriber.Config{Repository: base, ReplayPageLimit: 2})
				require.NoError(err)

				subCtx, cancel := context.WithCancel(context.Background())
				defer cancel()

				events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{SessionID: "s1", Replay: true})
				require.NoError(err)

				first := readEvent(t, events)
				assert.True(first.Replay)
				require.Len(first.Messages, 2)
				assert.Equal("m1", first.Messages[0].ID)
				assert.Equal("m2", first.Messages[1].ID)

				second := readEvent(t, events)
				assert.True(second.Replay)
				require.Len(second.Messages, 1)
				assert.Equal("m3", second.Messages[0].ID)
			},
		},

		"Replay without session ID should return error.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				base := memory.NewRepository()
				repo, err := subscriber.New(subscriber.Config{Repository: base})
				require.NoError(err)

				_, err = repo.Subscribe(context.Background(), subscriber.SubscribeOpts{Replay: true})
				assert.Error(err)
				assert.ErrorIs(err, pkgerrors.ErrNotValid)
			},
		},

		"Context cancellation should close the events channel.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				base := memory.NewRepository()
				repo, err := subscriber.New(subscriber.Config{Repository: base})
				require.NoError(err)

				subCtx, cancel := context.WithCancel(context.Background())
				events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{})
				require.NoError(err)

				cancel()

				select {
				case _, ok := <-events:
					assert.False(ok)
				case <-time.After(1 * time.Second):
					require.FailNow("timed out waiting for closed channel")
				}
			},
		},

		"Full buffer should drop oldest events and keep newest.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				ctx := context.Background()
				base := memory.NewRepository()
				require.NoError(base.CreateSession(ctx, model.Session{ID: "s1"}))

				repo, err := subscriber.New(subscriber.Config{Repository: base, BufferSize: 1})
				require.NoError(err)

				subCtx, cancel := context.WithCancel(context.Background())
				defer cancel()

				events, err := repo.Subscribe(subCtx, subscriber.SubscribeOpts{SessionID: "s1"})
				require.NoError(err)

				require.NoError(repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m1"}}))
				require.NoError(repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m2"}}))
				require.NoError(repo.StoreMessages(ctx, "s1", []model.Message{{ID: "m3"}}))

				event := readEvent(t, events)
				require.Len(event.Messages, 1)
				assert.Equal("m3", event.Messages[0].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

func readEvent(t *testing.T, events <-chan subscriber.MessageStoredEvent) subscriber.MessageStoredEvent {
	t.Helper()
	require := require.New(t)

	select {
	case event := <-events:
		return event
	case <-time.After(1 * time.Second):
		require.FailNow("timed out waiting for event")
		return subscriber.MessageStoredEvent{}
	}
}

func assertNoEvent(t *testing.T, events <-chan subscriber.MessageStoredEvent) {
	t.Helper()
	require := require.New(t)

	select {
	case event := <-events:
		require.FailNowf("unexpected event", "%+v", event)
	case <-time.After(100 * time.Millisecond):
	}
}

var _ store.MessageRepository = (*subscriber.Repository)(nil)
