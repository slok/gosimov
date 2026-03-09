package anthropic_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/llm/anthropic"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

func TestNewAPIKeyTokenSource(t *testing.T) {
	tests := map[string]struct {
		apiKey   string
		expToken string
		expErr   bool
		expErrIs error
	}{
		"Should return static token when API key is set.": {
			apiKey:   "sk-ant-test",
			expToken: "sk-ant-test",
		},
		"Should fail when API key is empty.": {
			apiKey:   "",
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			ts := anthropic.NewAPIKeyTokenSource(test.apiKey)
			tok, err := ts.Token(context.Background())

			if test.expErr {
				require.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				return
			}

			require.NoError(err)
			assert.Equal(test.expToken, tok)
		})
	}
}
