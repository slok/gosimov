package id

import (
	"crypto/rand"
	"fmt"
	"strings"

	"github.com/oklog/ulid/v2"
)

// NewULID generates a new ULID (Universally Unique Lexicographically Sortable Identifier).
//
// If prefix is not empty, the final ID format is: "<prefix>-<ulid>".
// If prefix already ends with '-', the trailing '-' is removed first.
func NewULID(prefix string) string {
	id := ulid.MustNew(ulid.Now(), rand.Reader).String()
	if prefix == "" {
		return id
	}

	p := strings.TrimSuffix(prefix, "-")
	if p == "" {
		return id
	}

	return fmt.Sprintf("%s-%s", p, id)
}
