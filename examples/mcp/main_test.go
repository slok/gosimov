package main

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeaderFlagsSet(t *testing.T) {
	tests := map[string]struct {
		values  []string
		expErr  bool
		expHead http.Header
	}{
		"Valid repeated headers should be parsed.": {
			values: []string{"Authorization: Bearer abc", "X-Test: one", "X-Test: two"},
			expHead: http.Header{
				"Authorization": []string{"Bearer abc"},
				"X-Test":        []string{"one", "two"},
			},
		},
		"Invalid header should fail.": {
			values: []string{"missing-separator"},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var flags headerFlags
			for _, value := range test.values {
				err := flags.Set(value)
				if test.expErr {
					require.Error(t, err)
					return
				}
				require.NoError(t, err)
			}

			assert.Equal(t, test.expHead, flags.Header())
		})
	}
}
