package subscriber

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/store"
)

const (
	defaultBufferSize  = 32
	defaultReplayLimit = 100
	maxBufferedBacklog = 4096
)

// Compile-time interface check.
var _ store.MessageRepository = (*Repository)(nil)

// Config configures a subscriber-enabled MessageRepository wrapper.
type Config struct {
	// Repository is the wrapped message repository (required).
	Repository store.MessageRepository
	// BufferSize is the default channel capacity for subscriptions.
	// 0 means default.
	BufferSize int
	// ReplayPageLimit is the page size used when replaying historical messages.
	// 0 means default.
	ReplayPageLimit int
}

func (c *Config) defaults() error {
	if c.Repository == nil {
		return fmt.Errorf("repository is required: %w", pkgerrors.ErrNotValid)
	}

	if c.BufferSize <= 0 {
		c.BufferSize = defaultBufferSize
	}

	if c.ReplayPageLimit <= 0 {
		c.ReplayPageLimit = defaultReplayLimit
	}

	return nil
}

// SubscribeOpts configures a subscription.
type SubscribeOpts struct {
	// SessionID filters events by session. Empty means all sessions.
	SessionID string
	// BufferSize overrides the wrapper default buffer size.
	// 0 means wrapper default.
	BufferSize int
	// Replay sends historical messages before live events.
	// Replay requires SessionID, because MessageRepository can only list
	// messages by session.
	Replay bool
	// ReplayCursor starts replay from a specific opaque cursor.
	// Empty means from the beginning.
	ReplayCursor string
}

// MessageStoredEvent is emitted after messages are successfully stored.
type MessageStoredEvent struct {
	SessionID string
	Messages  []model.Message
	StoredAt  time.Time
	Replay    bool
}

type subscriber struct {
	sessionID string
	ch        chan MessageStoredEvent
}

// Repository wraps a MessageRepository and publishes message-store events.
//
// Event delivery is best-effort and non-blocking. If a subscriber channel is full,
// the oldest event is dropped to make room for the newest one.
//
// Replay is a snapshot taken before the live subscription is registered. Messages
// stored between snapshot completion and live registration can be missed.
type Repository struct {
	repo            store.MessageRepository
	defaultBuffer   int
	replayPageLimit int
	now             func() time.Time

	mu          sync.RWMutex
	nextSubID   uint64
	subscribers map[uint64]subscriber
}

// New creates a subscriber-enabled MessageRepository wrapper.
func New(cfg Config) (*Repository, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &Repository{
		repo:            cfg.Repository,
		defaultBuffer:   cfg.BufferSize,
		replayPageLimit: cfg.ReplayPageLimit,
		now:             time.Now,
		subscribers:     map[uint64]subscriber{},
	}, nil
}

// StoreMessages stores messages and notifies subscribers on success.
func (r *Repository) StoreMessages(ctx context.Context, sessionID string, msgs []model.Message) error {
	err := r.repo.StoreMessages(ctx, sessionID, msgs)
	if err != nil {
		return err
	}

	if len(msgs) == 0 {
		return nil
	}

	r.notify(MessageStoredEvent{
		SessionID: sessionID,
		Messages:  cloneMessages(msgs),
		StoredAt:  r.now(),
	})

	return nil
}

// ListMessages delegates to the wrapped repository.
func (r *Repository) ListMessages(ctx context.Context, sessionID string, opts store.ListOpts) (*store.ListResult[model.Message], error) {
	return r.repo.ListMessages(ctx, sessionID, opts)
}

// Subscribe returns a stream of stored-message events.
//
// The returned channel is closed automatically when ctx is done.
func (r *Repository) Subscribe(ctx context.Context, opts SubscribeOpts) (<-chan MessageStoredEvent, error) {
	if opts.Replay && opts.SessionID == "" {
		return nil, fmt.Errorf("replay requires session id: %w", pkgerrors.ErrNotValid)
	}

	replayEvents, err := r.replayEvents(ctx, opts)
	if err != nil {
		return nil, err
	}

	bufferSize := r.defaultBuffer
	if opts.BufferSize > 0 {
		bufferSize = opts.BufferSize
	}

	backlog := len(replayEvents)
	if backlog > maxBufferedBacklog {
		backlog = maxBufferedBacklog
	}

	ch := make(chan MessageStoredEvent, bufferSize+backlog)

	for _, event := range replayEvents {
		ch <- event
	}

	r.mu.Lock()
	subID := r.nextSubID
	r.nextSubID++
	r.subscribers[subID] = subscriber{
		sessionID: opts.SessionID,
		ch:        ch,
	}
	r.mu.Unlock()

	go func() {
		<-ctx.Done()

		r.mu.Lock()
		sub, ok := r.subscribers[subID]
		if ok {
			delete(r.subscribers, subID)
			close(sub.ch)
		}
		r.mu.Unlock()
	}()

	return ch, nil
}

func (r *Repository) replayEvents(ctx context.Context, opts SubscribeOpts) ([]MessageStoredEvent, error) {
	if !opts.Replay {
		return nil, nil
	}

	cursor := opts.ReplayCursor
	result := []MessageStoredEvent{}

	for {
		list, err := r.repo.ListMessages(ctx, opts.SessionID, store.ListOpts{Cursor: cursor, Limit: r.replayPageLimit})
		if err != nil {
			return nil, fmt.Errorf("replaying messages for session %q: %w", opts.SessionID, err)
		}

		if len(list.Items) > 0 {
			result = append(result, MessageStoredEvent{
				SessionID: opts.SessionID,
				Messages:  cloneMessages(list.Items),
				StoredAt:  r.now(),
				Replay:    true,
			})
		}

		if list.NextCursor == "" {
			return result, nil
		}

		cursor = list.NextCursor
	}
}

func (r *Repository) notify(event MessageStoredEvent) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, sub := range r.subscribers {
		if sub.sessionID != "" && sub.sessionID != event.SessionID {
			continue
		}

		enqueueLatest(sub.ch, event)
	}
}

func enqueueLatest(ch chan MessageStoredEvent, event MessageStoredEvent) {
	select {
	case ch <- event:
		return
	default:
	}

	select {
	case <-ch:
	default:
	}

	select {
	case ch <- event:
	default:
	}
}

func cloneMessages(msgs []model.Message) []model.Message {
	if len(msgs) == 0 {
		return nil
	}

	cloned := make([]model.Message, len(msgs))
	copy(cloned, msgs)

	return cloned
}
