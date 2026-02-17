package id_test

import (
	"testing"

	"github.com/oklog/ulid/v2"

	"github.com/slok/gosimov/internal/utils/id"
)

func TestNewULID(t *testing.T) {
	tests := map[string]struct {
		check func(t *testing.T)
	}{
		"Generated ULID should not be empty.": {
			check: func(t *testing.T) {
				got := id.NewULID()
				if got == "" {
					t.Error("expected non-empty ULID")
				}
			},
		},

		"Generated ULID should be valid ULID format.": {
			check: func(t *testing.T) {
				got := id.NewULID()
				_, err := ulid.Parse(got)
				if err != nil {
					t.Errorf("expected valid ULID format, got %q: %v", got, err)
				}
			},
		},

		"Two generated ULIDs should be unique.": {
			check: func(t *testing.T) {
				a := id.NewULID()
				b := id.NewULID()
				if a == b {
					t.Errorf("expected unique ULIDs, got %q twice", a)
				}
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.check(t)
		})
	}
}
