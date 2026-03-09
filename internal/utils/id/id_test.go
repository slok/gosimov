package id_test

import (
	"testing"

	"github.com/oklog/ulid/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/internal/utils/id"
)

func TestNewULID(t *testing.T) {
	tests := map[string]struct {
		check func(t *testing.T)
	}{
		"Generated ULID should not be empty.": {
			check: func(t *testing.T) {
				assert := assert.New(t)

				got := id.NewULID()
				assert.NotEmpty(got)
			},
		},

		"Generated ULID should be valid ULID format.": {
			check: func(t *testing.T) {
				require := require.New(t)

				got := id.NewULID()
				_, err := ulid.Parse(got)
				require.NoError(err, "expected valid ULID format for %q", got)
			},
		},

		"Two generated ULIDs should be unique.": {
			check: func(t *testing.T) {
				assert := assert.New(t)

				a := id.NewULID()
				b := id.NewULID()
				assert.NotEqual(a, b)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.check(t)
		})
	}
}
