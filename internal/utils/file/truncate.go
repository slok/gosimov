package file

import (
	"fmt"
	"strings"
)

// TruncateOpts configures a [TruncateHead] call.
type TruncateOpts struct {
	// MaxBytes is the maximum number of bytes to keep. 0 means no limit.
	MaxBytes int
	// MaxLines is the maximum number of lines to keep. 0 means no limit.
	MaxLines int
}

// TruncateResult describes what happened during truncation.
type TruncateResult struct {
	// Truncated is true if the content was shortened.
	Truncated bool
	// OriginalBytes is the byte length of the input.
	OriginalBytes int
	// OriginalLines is the line count of the input.
	OriginalLines int
	// KeptBytes is the byte length of the output.
	KeptBytes int
	// KeptLines is the line count of the output.
	KeptLines int
}

// TruncateHead keeps the first portion of content that fits within the given limits.
// It truncates at line boundaries when possible — it will not split a line
// in half unless a single line exceeds MaxBytes.
//
// If both MaxBytes and MaxLines are 0, the content is returned unchanged.
func TruncateHead(content string, opts TruncateOpts) (string, TruncateResult) {
	originalBytes := len(content)
	originalLines := countLines(content)

	// No limits — return as-is.
	if opts.MaxBytes <= 0 && opts.MaxLines <= 0 {
		return content, TruncateResult{
			OriginalBytes: originalBytes,
			OriginalLines: originalLines,
			KeptBytes:     originalBytes,
			KeptLines:     originalLines,
		}
	}

	lines := strings.Split(content, "\n")
	var kept []string
	totalBytes := 0

	for i, line := range lines {
		// Check line limit.
		if opts.MaxLines > 0 && i >= opts.MaxLines {
			break
		}

		// Check byte limit. Account for the newline separator between lines.
		lineBytes := len(line)
		if i > 0 {
			lineBytes++ // for the \n between this line and the previous.
		}

		if opts.MaxBytes > 0 && totalBytes+lineBytes > opts.MaxBytes {
			// If this is the first line and it exceeds MaxBytes, hard-cut it.
			if i == 0 {
				kept = append(kept, line[:opts.MaxBytes])
			}

			break
		}

		kept = append(kept, line)
		totalBytes += lineBytes
	}

	result := strings.Join(kept, "\n")
	keptLines := countLines(result)

	return result, TruncateResult{
		Truncated:     len(result) < originalBytes,
		OriginalBytes: originalBytes,
		OriginalLines: originalLines,
		KeptBytes:     len(result),
		KeptLines:     keptLines,
	}
}

// TruncateTail keeps the last portion of content that fits within the given limits.
// It truncates at line boundaries when possible — it will not split a line
// in half unless a single line exceeds MaxBytes.
//
// If both MaxBytes and MaxLines are 0, the content is returned unchanged.
func TruncateTail(content string, opts TruncateOpts) (string, TruncateResult) {
	originalBytes := len(content)
	originalLines := countLines(content)

	// No limits — return as-is.
	if opts.MaxBytes <= 0 && opts.MaxLines <= 0 {
		return content, TruncateResult{
			OriginalBytes: originalBytes,
			OriginalLines: originalLines,
			KeptBytes:     originalBytes,
			KeptLines:     originalLines,
		}
	}

	lines := strings.Split(content, "\n")
	var kept []string
	totalBytes := 0

	// Walk backwards from the last line.
	for i := len(lines) - 1; i >= 0; i-- {
		// Check line limit.
		if opts.MaxLines > 0 && len(kept) >= opts.MaxLines {
			break
		}

		lineBytes := len(lines[i])
		if len(kept) > 0 {
			lineBytes++ // for the \n between this line and the next.
		}

		if opts.MaxBytes > 0 && totalBytes+lineBytes > opts.MaxBytes {
			// If this is the first kept line (i.e., the last line in the file)
			// and it exceeds MaxBytes, hard-cut from the end.
			if len(kept) == 0 {
				kept = append(kept, lines[i][len(lines[i])-opts.MaxBytes:])
			}
			break
		}

		kept = append(kept, lines[i])
		totalBytes += lineBytes
	}

	// Reverse kept to restore original order.
	for i, j := 0, len(kept)-1; i < j; i, j = i+1, j-1 {
		kept[i], kept[j] = kept[j], kept[i]
	}

	result := strings.Join(kept, "\n")
	keptLines := countLines(result)

	return result, TruncateResult{
		Truncated:     len(result) < originalBytes,
		OriginalBytes: originalBytes,
		OriginalLines: originalLines,
		KeptBytes:     len(result),
		KeptLines:     keptLines,
	}
}

// FormatSize returns a human-readable representation of a byte count.
//
//	FormatSize(1024)  => "1.0KB"
//	FormatSize(50000) => "48.8KB"
func FormatSize(bytes int) string {
	switch {
	case bytes >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(bytes)/float64(1<<30))
	case bytes >= 1<<20:
		return fmt.Sprintf("%.1fMB", float64(bytes)/float64(1<<20))
	case bytes >= 1<<10:
		return fmt.Sprintf("%.1fKB", float64(bytes)/float64(1<<10))
	default:
		return fmt.Sprintf("%dB", bytes)
	}
}

// countLines returns the number of lines in s. An empty string has 0 lines.
// A string with no newlines has 1 line.
func countLines(s string) int {
	if s == "" {
		return 0
	}

	return strings.Count(s, "\n") + 1
}
