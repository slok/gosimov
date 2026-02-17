package file_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/internal/utils/file"
)

func TestFormatUnifiedDiff(t *testing.T) {
	tests := map[string]struct {
		path         string
		oldContent   string
		newContent   string
		replacements []file.Replacement
		contextLines int
		expDiff      string
	}{
		"Simple single-line replacement.": {
			path:       "hello.txt",
			oldContent: "hello world",
			newContent: "hello gopher",
			replacements: []file.Replacement{
				{Offset: 6, OldLen: 5, NewLen: 6},
			},
			contextLines: 3,
			expDiff: `--- hello.txt
+++ hello.txt
@@ -1,1 +1,1 @@
-hello world
+hello gopher
`,
		},

		"Replacement with surrounding context lines.": {
			path:       "code.go",
			oldContent: "line1\nline2\nline3\nline4\nline5\nline6\nline7\n",
			newContent: "line1\nline2\nline3\nchanged\nline5\nline6\nline7\n",
			replacements: []file.Replacement{
				{Offset: 18, OldLen: 5, NewLen: 7}, // "line4" -> "changed" at line 4.
			},
			contextLines: 3,
			expDiff: `--- code.go
+++ code.go
@@ -1,7 +1,7 @@
 line1
 line2
 line3
-line4
+changed
 line5
 line6
 line7
`,
		},

		"Context limited to 1 line.": {
			path:       "file.txt",
			oldContent: "a\nb\nc\nd\ne\nf\ng\n",
			newContent: "a\nb\nc\nX\ne\nf\ng\n",
			replacements: []file.Replacement{
				{Offset: 6, OldLen: 1, NewLen: 1}, // "d" -> "X" at line 4.
			},
			contextLines: 1,
			expDiff: `--- file.txt
+++ file.txt
@@ -3,3 +3,3 @@
 c
-d
+X
 e
`,
		},

		"Multi-line old text replaced with single line.": {
			path:       "multi.txt",
			oldContent: "before\nold1\nold2\nold3\nafter\n",
			newContent: "before\nnew\nafter\n",
			replacements: []file.Replacement{
				{Offset: 7, OldLen: 14, NewLen: 3}, // "old1\nold2\nold3" -> "new".
			},
			contextLines: 3,
			expDiff: `--- multi.txt
+++ multi.txt
@@ -1,5 +1,3 @@
 before
-old1
-old2
-old3
+new
 after
`,
		},

		"Single line replaced with multiple lines.": {
			path:       "expand.txt",
			oldContent: "before\nold\nafter\n",
			newContent: "before\nnew1\nnew2\nnew3\nafter\n",
			replacements: []file.Replacement{
				{Offset: 7, OldLen: 3, NewLen: 13}, // "old" -> "new1\nnew2\nnew3".
			},
			contextLines: 3,
			expDiff: `--- expand.txt
+++ expand.txt
@@ -1,3 +1,5 @@
 before
-old
+new1
+new2
+new3
 after
`,
		},

		"Delete text (empty replacement).": {
			path:       "delete.txt",
			oldContent: "keep\nremove\nkeep\n",
			newContent: "keep\nkeep\n",
			replacements: []file.Replacement{
				{Offset: 5, OldLen: 7, NewLen: 0}, // "remove\n" -> "".
			},
			contextLines: 3,
			expDiff: `--- delete.txt
+++ delete.txt
@@ -1,3 +1,2 @@
 keep
-remove
 keep
`,
		},

		"Replace all — two separate hunks.": {
			path:       "replaceall.txt",
			oldContent: "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\n",
			newContent: "a\nB\nc\nd\ne\nf\ng\nh\ni\nJ\nk\n",
			replacements: []file.Replacement{
				{Offset: 2, OldLen: 1, NewLen: 1},  // "b" -> "B" at line 2.
				{Offset: 18, OldLen: 1, NewLen: 1}, // "j" -> "J" at line 10.
			},
			contextLines: 1,
			expDiff: `--- replaceall.txt
+++ replaceall.txt
@@ -1,3 +1,3 @@
 a
-b
+B
 c
@@ -9,3 +9,3 @@
 i
-j
+J
 k
`,
		},

		"Replace all — hunks merged when close together.": {
			path:       "close.txt",
			oldContent: "a\nb\nc\nd\ne\n",
			newContent: "a\nB\nc\nD\ne\n",
			replacements: []file.Replacement{
				{Offset: 2, OldLen: 1, NewLen: 1}, // "b" -> "B" at line 2.
				{Offset: 6, OldLen: 1, NewLen: 1}, // "d" -> "D" at line 4.
			},
			contextLines: 3,
			expDiff: `--- close.txt
+++ close.txt
@@ -1,5 +1,5 @@
 a
-b
+B
 c
-d
+D
 e
`,
		},

		"No replacements returns empty string.": {
			path:         "empty.txt",
			oldContent:   "hello",
			newContent:   "hello",
			replacements: nil,
			contextLines: 3,
			expDiff:      "",
		},

		"Replacement at very start of file.": {
			path:       "start.txt",
			oldContent: "old\nrest\n",
			newContent: "new\nrest\n",
			replacements: []file.Replacement{
				{Offset: 0, OldLen: 3, NewLen: 3},
			},
			contextLines: 3,
			expDiff: `--- start.txt
+++ start.txt
@@ -1,2 +1,2 @@
-old
+new
 rest
`,
		},

		"Replacement at very end of file.": {
			path:       "end.txt",
			oldContent: "first\nlast",
			newContent: "first\nend",
			replacements: []file.Replacement{
				{Offset: 6, OldLen: 4, NewLen: 3},
			},
			contextLines: 3,
			expDiff: `--- end.txt
+++ end.txt
@@ -1,2 +1,2 @@
 first
-last
+end
`,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := file.FormatUnifiedDiff(test.path, test.oldContent, test.newContent, test.replacements, test.contextLines)
			assert.Equal(t, test.expDiff, got)
		})
	}
}

func TestSplitLines(t *testing.T) {
	tests := map[string]struct {
		input string
		exp   []string
	}{
		"Empty string.": {
			input: "",
			exp:   nil,
		},

		"Single line without newline.": {
			input: "hello",
			exp:   []string{"hello"},
		},

		"Single line with newline.": {
			input: "hello\n",
			exp:   []string{"hello"},
		},

		"Multiple lines.": {
			input: "a\nb\nc\n",
			exp:   []string{"a", "b", "c"},
		},

		"Multiple lines without trailing newline.": {
			input: "a\nb\nc",
			exp:   []string{"a", "b", "c"},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := file.SplitLines(test.input)
			assert.Equal(t, test.exp, got)
		})
	}
}

func TestByteOffsetToLine(t *testing.T) {
	tests := map[string]struct {
		content string
		offset  int
		expLine int
	}{
		"First character.": {
			content: "hello\nworld\n",
			offset:  0,
			expLine: 0,
		},

		"Start of second line.": {
			content: "hello\nworld\n",
			offset:  6,
			expLine: 1,
		},

		"Middle of second line.": {
			content: "hello\nworld\n",
			offset:  8,
			expLine: 1,
		},

		"Negative offset.": {
			content: "hello\n",
			offset:  -1,
			expLine: 0,
		},

		"Offset beyond content.": {
			content: "hello\nworld\n",
			offset:  100,
			expLine: 1, // Clamped to last byte (index 11 = '\n'), still on line 1.
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := file.ByteOffsetToLine(test.content, test.offset)
			assert.Equal(t, test.expLine, got)
		})
	}
}
