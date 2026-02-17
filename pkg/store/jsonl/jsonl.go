// Package jsonl implements [store.SessionRepository] and [store.MessageRepository]
// using JSONL files on disk.
//
// Each session is stored as a single self-contained file named <session_id>.jsonl.
// The first line is the session header, followed by one line per message (append-only).
// Deleting a session file removes all its data.
package jsonl

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/slok/gosimov/internal/utils/file"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/store"
)

// Compile-time interface checks.
var (
	_ store.SessionRepository = (*Repository)(nil)
	_ store.MessageRepository = (*Repository)(nil)
)

// Config configures the JSONL store.
type Config struct {
	// Dir is the directory where session files are stored (required).
	// Created automatically if it doesn't exist.
	Dir string
	// FS is the filesystem implementation (optional).
	// Defaults to OS filesystem rooted at Dir.
	FS file.ReadWriteFS
}

func (c *Config) defaults() error {
	if c.Dir == "" {
		return fmt.Errorf("dir is required: %w", pkgerrors.ErrNotValid)
	}

	if c.FS == nil {
		c.FS = file.NewOSReadWriteFS(c.Dir)
	}

	return nil
}

// Repository is a JSONL-based implementation of [store.SessionRepository] and
// [store.MessageRepository].
//
// Each session is a single file: <dir>/<session_id>.jsonl.
// The first line is the session header, remaining lines are messages (append-only).
//
// Safe for concurrent use within the same process.
type Repository struct {
	mu sync.RWMutex
	fs file.ReadWriteFS
}

// New creates a new JSONL store repository.
// The directory is created if it doesn't exist.
func New(cfg Config) (*Repository, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid jsonl store config: %w", err)
	}

	// Ensure the root directory exists (MkdirAll on "." is a no-op for osReadWriteFS
	// since the root was already validated, but custom FS implementations may need it).
	if err := cfg.FS.MkdirAll("."); err != nil {
		return nil, fmt.Errorf("creating store directory: %w", err)
	}

	return &Repository{
		fs: cfg.FS,
	}, nil
}

// CreateSession stores a new session by creating a JSONL file with the session header.
// Returns [pkgerrors.ErrAlreadyExists] if the file already exists.
func (r *Repository) CreateSession(_ context.Context, session model.Session) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	path := r.sessionPath(session.ID)

	// Check if file already exists.
	if _, err := r.fs.Stat(path); err == nil {
		return fmt.Errorf("session %q: %w", session.ID, pkgerrors.ErrAlreadyExists)
	}

	// Write session header as the first line.
	line := sessionToLine(session)
	data, err := json.Marshal(line)
	if err != nil {
		return fmt.Errorf("marshaling session line: %w", err)
	}
	data = append(data, '\n')

	if err := r.fs.WriteFile(path, data); err != nil {
		return fmt.Errorf("writing session file: %w", err)
	}

	return nil
}

// GetSession retrieves a session by reading the first line of its JSONL file.
// Returns [pkgerrors.ErrNotFound] if the file does not exist.
func (r *Repository) GetSession(_ context.Context, id string) (*model.Session, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	sl, err := r.readSessionLine(id)
	if err != nil {
		return nil, err
	}

	s := lineToSession(sl)
	return &s, nil
}

// ListSessions returns sessions ordered by creation time (newest first).
// Scans all .jsonl files in the directory and reads the first line of each.
func (r *Repository) ListSessions(_ context.Context, opts store.ListOpts) (*store.ListResult[model.Session], error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	entries, err := r.fs.ReadDir(".")
	if err != nil {
		return nil, fmt.Errorf("reading store directory: %w", err)
	}

	var sessions []model.Session
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
			continue
		}

		// Extract session ID from filename.
		id := strings.TrimSuffix(entry.Name(), ".jsonl")

		sl, err := r.readSessionLineFromFile(entry.Name())
		if err != nil {
			continue // Skip corrupt files.
		}

		// Sanity check: filename should match session ID.
		if sl.ID != id {
			continue
		}

		sessions = append(sessions, lineToSession(sl))
	}

	// Sort newest first by CreatedAt.
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].CreatedAt.After(sessions[j].CreatedAt)
	})

	return paginate(sessions, opts), nil
}

// StoreMessages appends messages to a session's JSONL file.
// Returns [pkgerrors.ErrNotFound] if the session file does not exist.
func (r *Repository) StoreMessages(_ context.Context, sessionID string, msgs []model.Message) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	path := r.sessionPath(sessionID)

	// Verify session exists.
	if _, err := r.fs.Stat(path); err != nil {
		return fmt.Errorf("session %q: %w", sessionID, pkgerrors.ErrNotFound)
	}

	if len(msgs) == 0 {
		return nil
	}

	// Build all lines in a single buffer for one append operation.
	var buf bytes.Buffer
	for _, msg := range msgs {
		line := messageToLine(msg)
		data, err := json.Marshal(line)
		if err != nil {
			return fmt.Errorf("marshaling message %q: %w", msg.ID, err)
		}
		buf.Write(data)
		buf.WriteByte('\n')
	}

	if err := r.fs.AppendFile(path, buf.Bytes()); err != nil {
		return fmt.Errorf("appending messages to session %q: %w", sessionID, err)
	}

	return nil
}

// ListMessages returns messages for a session in insertion order.
// Returns [pkgerrors.ErrNotFound] if the session file does not exist.
func (r *Repository) ListMessages(_ context.Context, sessionID string, opts store.ListOpts) (*store.ListResult[model.Message], error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	path := r.sessionPath(sessionID)

	data, err := r.fs.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("session %q: %w", sessionID, pkgerrors.ErrNotFound)
		}
		return nil, fmt.Errorf("reading session file %q: %w", sessionID, err)
	}

	messages, err := parseMessages(data)
	if err != nil {
		return nil, fmt.Errorf("parsing messages for session %q: %w", sessionID, err)
	}

	return paginate(messages, opts), nil
}

// --- Internal helpers ---

// sessionPath returns the file path for a session (relative to the FS root).
func (r *Repository) sessionPath(id string) string {
	return id + ".jsonl"
}

// readSessionLine reads and parses the session header from a session file by ID.
func (r *Repository) readSessionLine(id string) (sessionLine, error) {
	return r.readSessionLineFromFile(r.sessionPath(id))
}

// readSessionLineFromFile reads and parses the session header (first line) from a file path.
func (r *Repository) readSessionLineFromFile(path string) (sessionLine, error) {
	data, err := r.fs.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return sessionLine{}, fmt.Errorf("session file: %w", pkgerrors.ErrNotFound)
		}
		return sessionLine{}, fmt.Errorf("reading session file: %w", err)
	}

	// Read only the first line.
	scanner := bufio.NewScanner(bytes.NewReader(data))
	if !scanner.Scan() {
		return sessionLine{}, fmt.Errorf("empty session file: %w", pkgerrors.ErrNotValid)
	}

	var header lineHeader
	if err := json.Unmarshal(scanner.Bytes(), &header); err != nil {
		return sessionLine{}, fmt.Errorf("parsing line header: %w", err)
	}

	if header.Type != lineTypeSession {
		return sessionLine{}, fmt.Errorf("first line is not a session header: %w", pkgerrors.ErrNotValid)
	}

	var sl sessionLine
	if err := json.Unmarshal(scanner.Bytes(), &sl); err != nil {
		return sessionLine{}, fmt.Errorf("parsing session line: %w", err)
	}

	return sl, nil
}

// parseMessages parses all message lines from file data (skipping the session header).
func parseMessages(data []byte) ([]model.Message, error) {
	var messages []model.Message

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var header lineHeader
		if err := json.Unmarshal(line, &header); err != nil {
			continue // Skip unparseable lines.
		}

		if header.Type != lineTypeMessage {
			continue // Skip non-message lines (session header, etc.).
		}

		var ml messageLine
		if err := json.Unmarshal(line, &ml); err != nil {
			continue // Skip corrupt message lines.
		}

		messages = append(messages, lineToMessage(ml))
	}

	return messages, nil
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
