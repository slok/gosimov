package file

import (
	"fmt"
	"path/filepath"
	"strings"
)

// SanitizePath cleans and validates a path to ensure it stays within the working
// directory. It rejects absolute paths and paths that escape via "..".
//
// The returned path uses forward slashes for [io/fs.FS] compatibility.
func SanitizePath(p string) (string, error) {
	// Reject absolute paths.
	if filepath.IsAbs(p) {
		return "", fmt.Errorf("absolute paths are not allowed: %s", p)
	}

	// Clean the path to resolve any . or .. components.
	cleaned := filepath.Clean(p)

	// Reject paths that escape the working directory.
	if cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes working directory: %s", p)
	}

	// Convert to forward slashes for fs.FS compatibility.
	return filepath.ToSlash(cleaned), nil
}
