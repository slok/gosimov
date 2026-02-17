package auth_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"
)

func TestFileCredentialsStoreRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "auth.json")

	store, err := llmauth.NewFileCredentialsStore(path)
	if err != nil {
		t.Fatalf("unexpected error creating store: %v", err)
	}

	creds := llmauth.OAuthCredentials{
		AccessToken:  "a1",
		RefreshToken: "r1",
		Expiry:       time.Now().Add(1 * time.Hour).UTC().Truncate(time.Second),
	}

	if err := store.Save(context.Background(), "test", creds); err != nil {
		t.Fatalf("unexpected save error: %v", err)
	}

	got, err := store.Load(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected load error: %v", err)
	}
	if got == nil {
		t.Fatal("expected credentials, got nil")
	}

	if got.AccessToken != creds.AccessToken || got.RefreshToken != creds.RefreshToken || !got.Expiry.Equal(creds.Expiry) {
		t.Fatalf("credentials mismatch: got=%+v want=%+v", *got, creds)
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat auth file: %v", err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("expected mode 0600, got %o", fi.Mode().Perm())
	}
}
