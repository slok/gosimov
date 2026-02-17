package file_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/internal/utils/file"
)

func TestSanitizePath(t *testing.T) {
	tests := map[string]struct {
		path    string
		expPath string
		expErr  bool
	}{
		"Simple relative path.": {
			path:    "foo/bar",
			expPath: "foo/bar",
		},

		"Dot path.": {
			path:    ".",
			expPath: ".",
		},

		"Dot-slash prefix.": {
			path:    "./src",
			expPath: "src",
		},

		"Relative path with .. that stays inside.": {
			path:    "sub/..",
			expPath: ".",
		},

		"Nested relative path with .. that stays inside.": {
			path:    "a/b/../c",
			expPath: "a/c",
		},

		"Absolute path should be rejected.": {
			path:   "/etc/passwd",
			expErr: true,
		},

		"Dot-dot alone should be rejected.": {
			path:   "..",
			expErr: true,
		},

		"Path traversal with .. should be rejected.": {
			path:   "../../etc",
			expErr: true,
		},

		"Hidden path traversal should be rejected.": {
			path:   "foo/../../..",
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			result, err := file.SanitizePath(test.path)

			if test.expErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expPath, result)
			}
		})
	}
}
