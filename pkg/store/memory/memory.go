package memory

import (
	"context"
	"fmt"
	"strconv"
	"sync"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/store"
)

// Compile-time interface checks.
var (
	_ store.SessionRepository = (*Repository)(nil)
	_ store.MessageRepository = (*Repository)(nil)
)

// Repository is an in-memory implementation of [store.SessionRepository] and
// [store.MessageRepository]. All data is lost when the process exits.
//
// Safe for concurrent use.
type Repository struct {
	mu       sync.RWMutex
	sessions map[string]model.Session
	messages map[string][]model.Message
	// sessionOrder tracks insertion order for ListSessions (newest first).
	sessionOrder []string
}

// NewRepository creates a new in-memory repository.
func NewRepository() *Repository {
	return &Repository{
		sessions: make(map[string]model.Session),
		messages: make(map[string][]model.Message),
	}
}

// CreateSession stores a new session. Returns [pkgerrors.ErrAlreadyExists]
// if a session with the same ID already exists.
func (r *Repository) CreateSession(_ context.Context, session model.Session) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.sessions[session.ID]; ok {
		return fmt.Errorf("session %q: %w", session.ID, pkgerrors.ErrAlreadyExists)
	}

	r.sessions[session.ID] = session
	r.messages[session.ID] = nil
	r.sessionOrder = append(r.sessionOrder, session.ID)

	return nil
}

// GetSession retrieves a session by ID. Returns [pkgerrors.ErrNotFound]
// if the session does not exist.
func (r *Repository) GetSession(_ context.Context, id string) (*model.Session, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	s, ok := r.sessions[id]
	if !ok {
		return nil, fmt.Errorf("session %q: %w", id, pkgerrors.ErrNotFound)
	}

	return &s, nil
}

// ListSessions returns sessions ordered by creation time (newest first).
func (r *Repository) ListSessions(_ context.Context, opts store.ListOpts) (*store.ListResult[model.Session], error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Build sorted list (newest first = reverse insertion order).
	sorted := make([]model.Session, 0, len(r.sessionOrder))
	for i := len(r.sessionOrder) - 1; i >= 0; i-- {
		sorted = append(sorted, r.sessions[r.sessionOrder[i]])
	}

	return paginate(sorted, opts), nil
}

// StoreMessages stores messages in a session's conversation history.
// Returns [pkgerrors.ErrNotFound] if the session does not exist.
func (r *Repository) StoreMessages(_ context.Context, sessionID string, msgs []model.Message) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.sessions[sessionID]; !ok {
		return fmt.Errorf("session %q: %w", sessionID, pkgerrors.ErrNotFound)
	}

	r.messages[sessionID] = append(r.messages[sessionID], msgs...)

	return nil
}

// ListMessages returns messages for a session in insertion order.
// Returns [pkgerrors.ErrNotFound] if the session does not exist.
func (r *Repository) ListMessages(_ context.Context, sessionID string, opts store.ListOpts) (*store.ListResult[model.Message], error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	msgs, ok := r.messages[sessionID]
	if !ok {
		return nil, fmt.Errorf("session %q: %w", sessionID, pkgerrors.ErrNotFound)
	}

	return paginate(msgs, opts), nil
}

// paginate applies cursor-based pagination to a slice.
// Cursor is an index encoded as a decimal string. Empty cursor = start from 0.
// Limit 0 = return all remaining items.
func paginate[T any](items []T, opts store.ListOpts) *store.ListResult[T] {
	start := 0
	if opts.Cursor != "" {
		parsed, err := strconv.Atoi(opts.Cursor)
		if err == nil && parsed >= 0 {
			start = parsed
		}
	}

	if start >= len(items) {
		return &store.ListResult[T]{}
	}

	remaining := items[start:]

	if opts.Limit > 0 && opts.Limit < len(remaining) {
		result := remaining[:opts.Limit]
		nextCursor := strconv.Itoa(start + opts.Limit)

		return &store.ListResult[T]{
			Items:      result,
			NextCursor: nextCursor,
		}
	}

	return &store.ListResult[T]{
		Items: remaining,
	}
}
