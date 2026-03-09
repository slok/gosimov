package auth_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"
)

func TestFileCredentialsStoreRoundTrip(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Save and load should return the same credentials.": {
			run: func(t *testing.T) {
				assert := assert.New(t)
				require := require.New(t)

				tmpDir := t.TempDir()
				path := filepath.Join(tmpDir, "auth.json")

				store, err := llmauth.NewFileCredentialsStore(path)
				require.NoError(err)

				creds := llmauth.OAuthCredentials{
					AccessToken:  "a1",
					RefreshToken: "r1",
					Expiry:       time.Now().Add(1 * time.Hour).UTC().Truncate(time.Second),
				}

				err = store.Save(context.Background(), "test", creds)
				require.NoError(err)

				got, err := store.Load(context.Background(), "test")
				require.NoError(err)
				require.NotNil(got)

				assert.Equal(creds.AccessToken, got.AccessToken)
				assert.Equal(creds.RefreshToken, got.RefreshToken)
				assert.True(got.Expiry.Equal(creds.Expiry))

				fi, err := os.Stat(path)
				require.NoError(err)
				assert.Equal(os.FileMode(0o600), fi.Mode().Perm())
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}
