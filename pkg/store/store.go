package store

import (
	"context"

	"github.com/slok/gosimov/pkg/model"
)

// ListOpts configures cursor-based pagination for list operations.
//
// The Cursor is an opaque string whose format is defined by each
// implementation (e.g., an index, a timestamp, a ULID). An empty
// cursor starts from the beginning.
type ListOpts struct {
	// Cursor is the opaque pagination cursor. Empty = start from the beginning.
	Cursor string
	// Limit is the maximum number of items to return. 0 = no limit (return all).
	Limit int
}

// ListResult is the generic result of a paginated list operation.
type ListResult[T any] struct {
	// Items is the page of results.
	Items []T
	// NextCursor is the cursor for the next page. Empty = no more items.
	NextCursor string
}

// SessionRepository persists session identity.
type SessionRepository interface {
	// CreateSession stores a new session. Returns [pkgerrors.ErrAlreadyExists]
	// if a session with the same ID already exists.
	CreateSession(ctx context.Context, session model.Session) error
	// GetSession retrieves a session by ID. Returns [pkgerrors.ErrNotFound]
	// if the session does not exist.
	GetSession(ctx context.Context, id string) (*model.Session, error)
	// ListSessions returns sessions ordered by creation time (newest first).
	ListSessions(ctx context.Context, opts ListOpts) (*ListResult[model.Session], error)
}

// MessageRepository persists messages within sessions.
type MessageRepository interface {
	// StoreMessages stores messages in a session's conversation history.
	// Returns [pkgerrors.ErrNotFound] if the session does not exist.
	StoreMessages(ctx context.Context, sessionID string, msgs []model.Message) error
	// ListMessages returns messages for a session in insertion order.
	// Returns [pkgerrors.ErrNotFound] if the session does not exist.
	ListMessages(ctx context.Context, sessionID string, opts ListOpts) (*ListResult[model.Message], error)
}
