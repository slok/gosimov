package id

import (
	"crypto/rand"

	"github.com/oklog/ulid/v2"
)

// NewULID generates a new ULID (Universally Unique Lexicographically Sortable Identifier).
func NewULID() string {
	return ulid.MustNew(ulid.Now(), rand.Reader).String()
}
