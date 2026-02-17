package anthropic_test

import (
	"context"
	"errors"
	"testing"

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
			ts := anthropic.NewAPIKeyTokenSource(test.apiKey)
			tok, err := ts.Token(context.Background())

			if test.expErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if test.expErrIs != nil && !errors.Is(err, test.expErrIs) {
					t.Fatalf("expected wrapped error %v, got %v", test.expErrIs, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tok != test.expToken {
				t.Fatalf("expected token %q, got %q", test.expToken, tok)
			}
		})
	}
}
