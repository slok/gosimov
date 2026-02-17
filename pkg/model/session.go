package model

import "time"

// Session identifies a multi-turn conversation. It is a domain entity with
// identity — messages, usage, and other state are associated to a session via
// its ID but are not stored inside this struct.
type Session struct {
	// ID uniquely identifies this session (ULID).
	ID string
	// CreatedAt is the time the session was created.
	CreatedAt time.Time
}
