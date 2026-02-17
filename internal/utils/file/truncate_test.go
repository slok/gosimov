package file_test

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/internal/utils/file"
)

func TestTruncateHead(t *testing.T) {
	tests := map[string]struct {
		content   string
		opts      file.TruncateOpts
		expOutput string
		expResult file.TruncateResult
	}{
		"No limits should return content unchanged.": {
			content:   "line1\nline2\nline3",
			opts:      file.TruncateOpts{},
			expOutput: "line1\nline2\nline3",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 17,
				OriginalLines: 3,
				KeptBytes:     17,
				KeptLines:     3,
			},
		},

		"Empty content should return empty.": {
			content:   "",
			opts:      file.TruncateOpts{MaxBytes: 100, MaxLines: 10},
			expOutput: "",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 0,
				OriginalLines: 0,
				KeptBytes:     0,
				KeptLines:     0,
			},
		},

		"Content within both limits should not be truncated.": {
			content:   "short",
			opts:      file.TruncateOpts{MaxBytes: 100, MaxLines: 10},
			expOutput: "short",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 5,
				OriginalLines: 1,
				KeptBytes:     5,
				KeptLines:     1,
			},
		},

		"Line limit should truncate at line boundary.": {
			content:   "line1\nline2\nline3\nline4",
			opts:      file.TruncateOpts{MaxLines: 2},
			expOutput: "line1\nline2",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 23,
				OriginalLines: 4,
				KeptBytes:     11,
				KeptLines:     2,
			},
		},

		"Byte limit should truncate at line boundary.": {
			content: "aaa\nbbb\nccc",
			opts:    file.TruncateOpts{MaxBytes: 8},
			// "aaa\nbbb" = 7 bytes, "aaa\nbbb\nccc" = 11, next line would push to 11.
			expOutput: "aaa\nbbb",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 11,
				OriginalLines: 3,
				KeptBytes:     7,
				KeptLines:     2,
			},
		},

		"Byte limit should hard-cut first line if it exceeds limit.": {
			content:   "abcdefghij",
			opts:      file.TruncateOpts{MaxBytes: 5},
			expOutput: "abcde",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 10,
				OriginalLines: 1,
				KeptBytes:     5,
				KeptLines:     1,
			},
		},

		"Both limits with line limit winning.": {
			content:   "a\nb\nc\nd\ne",
			opts:      file.TruncateOpts{MaxBytes: 100, MaxLines: 3},
			expOutput: "a\nb\nc",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 9,
				OriginalLines: 5,
				KeptBytes:     5,
				KeptLines:     3,
			},
		},

		"Both limits with byte limit winning.": {
			content: "aaaa\nbbbb\ncccc",
			opts:    file.TruncateOpts{MaxBytes: 6, MaxLines: 100},
			// "aaaa" = 4 bytes, "aaaa\nbbbb" = 9 bytes > 6, so only first line.
			expOutput: "aaaa",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 14,
				OriginalLines: 3,
				KeptBytes:     4,
				KeptLines:     1,
			},
		},

		"Exact byte boundary should not truncate.": {
			content: "abc\ndef",
			opts:    file.TruncateOpts{MaxBytes: 7},
			// "abc\ndef" = exactly 7 bytes.
			expOutput: "abc\ndef",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 7,
				OriginalLines: 2,
				KeptBytes:     7,
				KeptLines:     2,
			},
		},

		"Exact line boundary should not truncate.": {
			content:   "a\nb\nc",
			opts:      file.TruncateOpts{MaxLines: 3},
			expOutput: "a\nb\nc",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 5,
				OriginalLines: 3,
				KeptBytes:     5,
				KeptLines:     3,
			},
		},

		"Single line within byte limit.": {
			content:   "hello world",
			opts:      file.TruncateOpts{MaxBytes: 100},
			expOutput: "hello world",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 11,
				OriginalLines: 1,
				KeptBytes:     11,
				KeptLines:     1,
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			output, result := file.TruncateHead(test.content, test.opts)

			assert.Equal(t, test.expOutput, output)
			assert.Equal(t, test.expResult, result)
		})
	}
}

func TestFormatSize(t *testing.T) {
	tests := map[string]struct {
		bytes  int
		expStr string
	}{
		"Bytes.":     {bytes: 512, expStr: "512B"},
		"Kilobytes.": {bytes: 1024, expStr: "1.0KB"},
		"50KB.":      {bytes: 50 * 1024, expStr: "50.0KB"},
		"Megabytes.": {bytes: 1024 * 1024, expStr: "1.0MB"},
		"Gigabytes.": {bytes: 1024 * 1024 * 1024, expStr: "1.0GB"},
		"Zero.":      {bytes: 0, expStr: "0B"},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.expStr, file.FormatSize(test.bytes))
		})
	}
}

func TestTruncateTail(t *testing.T) {
	tests := map[string]struct {
		content   string
		opts      file.TruncateOpts
		expOutput string
		expResult file.TruncateResult
	}{
		"No limits should return content unchanged.": {
			content:   "line1\nline2\nline3",
			opts:      file.TruncateOpts{},
			expOutput: "line1\nline2\nline3",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 17,
				OriginalLines: 3,
				KeptBytes:     17,
				KeptLines:     3,
			},
		},

		"Empty content should return empty.": {
			content:   "",
			opts:      file.TruncateOpts{MaxBytes: 100, MaxLines: 10},
			expOutput: "",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 0,
				OriginalLines: 0,
				KeptBytes:     0,
				KeptLines:     0,
			},
		},

		"Content within both limits should not be truncated.": {
			content:   "short",
			opts:      file.TruncateOpts{MaxBytes: 100, MaxLines: 10},
			expOutput: "short",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 5,
				OriginalLines: 1,
				KeptBytes:     5,
				KeptLines:     1,
			},
		},

		"Line limit should keep the last lines.": {
			content:   "line1\nline2\nline3\nline4",
			opts:      file.TruncateOpts{MaxLines: 2},
			expOutput: "line3\nline4",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 23,
				OriginalLines: 4,
				KeptBytes:     11,
				KeptLines:     2,
			},
		},

		"Byte limit should keep the last lines that fit.": {
			content: "aaa\nbbb\nccc",
			opts:    file.TruncateOpts{MaxBytes: 8},
			// "bbb\nccc" = 7 bytes fits, adding "aaa\n" would be 11 > 8.
			expOutput: "bbb\nccc",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 11,
				OriginalLines: 3,
				KeptBytes:     7,
				KeptLines:     2,
			},
		},

		"Byte limit should hard-cut last line if it exceeds limit.": {
			content:   "abcdefghij",
			opts:      file.TruncateOpts{MaxBytes: 5},
			expOutput: "fghij",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 10,
				OriginalLines: 1,
				KeptBytes:     5,
				KeptLines:     1,
			},
		},

		"Both limits with line limit winning.": {
			content:   "a\nb\nc\nd\ne",
			opts:      file.TruncateOpts{MaxBytes: 100, MaxLines: 3},
			expOutput: "c\nd\ne",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 9,
				OriginalLines: 5,
				KeptBytes:     5,
				KeptLines:     3,
			},
		},

		"Both limits with byte limit winning.": {
			content: "aaaa\nbbbb\ncccc",
			opts:    file.TruncateOpts{MaxBytes: 6, MaxLines: 100},
			// "cccc" = 4 bytes, "bbbb\ncccc" = 9 > 6, so only last line.
			expOutput: "cccc",
			expResult: file.TruncateResult{
				Truncated:     true,
				OriginalBytes: 14,
				OriginalLines: 3,
				KeptBytes:     4,
				KeptLines:     1,
			},
		},

		"Exact byte boundary should not truncate.": {
			content: "abc\ndef",
			opts:    file.TruncateOpts{MaxBytes: 7},
			// "abc\ndef" = exactly 7 bytes.
			expOutput: "abc\ndef",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 7,
				OriginalLines: 2,
				KeptBytes:     7,
				KeptLines:     2,
			},
		},

		"Exact line boundary should not truncate.": {
			content:   "a\nb\nc",
			opts:      file.TruncateOpts{MaxLines: 3},
			expOutput: "a\nb\nc",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 5,
				OriginalLines: 3,
				KeptBytes:     5,
				KeptLines:     3,
			},
		},

		"Single line within byte limit.": {
			content:   "hello world",
			opts:      file.TruncateOpts{MaxBytes: 100},
			expOutput: "hello world",
			expResult: file.TruncateResult{
				Truncated:     false,
				OriginalBytes: 11,
				OriginalLines: 1,
				KeptBytes:     11,
				KeptLines:     1,
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			output, result := file.TruncateTail(test.content, test.opts)

			assert.Equal(t, test.expOutput, output)
			assert.Equal(t, test.expResult, result)
		})
	}
}

func TestTruncateTailLargeContent(t *testing.T) {
	// Generate content that exceeds the byte limit.
	lines := make([]string, 1000)
	for i := range lines {
		lines[i] = strings.Repeat("x", 100)
	}
	content := strings.Join(lines, "\n")

	output, result := file.TruncateTail(content, file.TruncateOpts{MaxBytes: 50 * 1024})

	assert.True(t, result.Truncated)
	assert.LessOrEqual(t, result.KeptBytes, 50*1024)
	assert.Less(t, len(output), len(content))
	// Should contain the last line.
	assert.True(t, strings.HasSuffix(output, strings.Repeat("x", 100)))
}

func TestTruncateHeadLargeContent(t *testing.T) {
	// Generate content that exceeds the byte limit.
	lines := make([]string, 1000)
	for i := range lines {
		lines[i] = strings.Repeat("x", 100)
	}
	content := strings.Join(lines, "\n")

	output, result := file.TruncateHead(content, file.TruncateOpts{MaxBytes: 50 * 1024})

	assert.True(t, result.Truncated)
	assert.LessOrEqual(t, result.KeptBytes, 50*1024)
	assert.Less(t, len(output), len(content))
}
