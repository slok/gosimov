package file

import (
	"fmt"
	"strings"
)

// DefaultContextLines is the number of unchanged lines shown around each change in a unified diff.
const DefaultContextLines = 3

// Replacement describes a single text replacement by byte offset in the original content.
type Replacement struct {
	// Offset is the byte position in the original content where old text starts.
	Offset int
	// OldLen is the byte length of the text being replaced.
	OldLen int
	// NewLen is the byte length of the replacement text.
	NewLen int
}

// FormatUnifiedDiff produces a unified diff string for the given replacements.
//
// Both oldContent and newContent must be the full file content (before and after
// all replacements). The replacements slice describes each replacement by byte
// offset in oldContent. Replacements must be sorted by offset and non-overlapping.
//
// The path is used in the --- / +++ header lines.
// contextLines controls how many unchanged lines surround each hunk (use [DefaultContextLines] for standard behavior).
func FormatUnifiedDiff(path string, oldContent, newContent string, replacements []Replacement, contextLines int) string {
	if len(replacements) == 0 {
		return ""
	}

	oldLines := SplitLines(oldContent)
	newLines := SplitLines(newContent)

	// Convert byte-level replacements to line-level changes.
	changes := computeChanges(oldContent, newContent, replacements)

	// Group changes into hunks (merging those whose context would overlap).
	groups := groupChanges(changes, contextLines, len(oldLines), len(newLines))

	var b strings.Builder
	fmt.Fprintf(&b, "--- %s\n", path)
	fmt.Fprintf(&b, "+++ %s\n", path)

	for _, g := range groups {
		h := buildHunk(oldLines, newLines, g, contextLines)
		fmt.Fprintf(&b, "@@ -%d,%d +%d,%d @@\n", h.oldStart, h.oldCount, h.newStart, h.newCount)
		for _, line := range h.lines {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}

	return b.String()
}

// hunk represents a single hunk in a unified diff.
type hunk struct {
	oldStart int      // 1-indexed start line in old file.
	oldCount int      // Number of lines from old file in hunk.
	newStart int      // 1-indexed start line in new file.
	newCount int      // Number of lines from new file in hunk.
	lines    []string // Diff lines (prefixed with " ", "-", or "+").
}

// change describes one replacement in terms of line ranges.
type change struct {
	oldStart int  // 0-indexed first affected line in old content.
	oldEnd   int  // 0-indexed last affected line in old content (inclusive).
	newStart int  // 0-indexed first affected line in new content.
	newEnd   int  // 0-indexed last affected line in new content (inclusive).
	inserts  bool // Whether there are new lines to show (false for pure full-line deletions).
}

// computeChanges converts byte-offset replacements to line-level changes.
func computeChanges(oldContent, newContent string, replacements []Replacement) []change {
	changes := make([]change, 0, len(replacements))

	shift := 0
	for _, rep := range replacements {
		c := change{}

		// Old side: the line range covered by the old text.
		c.oldStart = ByteOffsetToLine(oldContent, rep.Offset)
		if rep.OldLen > 0 {
			c.oldEnd = ByteOffsetToLine(oldContent, rep.Offset+rep.OldLen-1)
		} else {
			c.oldEnd = c.oldStart
		}

		// New side: the line range covered by the new text.
		newOffset := rep.Offset + shift
		c.newStart = ByteOffsetToLine(newContent, newOffset)
		if rep.NewLen > 0 {
			c.newEnd = ByteOffsetToLine(newContent, newOffset+rep.NewLen-1)
			c.inserts = true
		} else {
			// Pure deletion. Check if the old text spans complete lines — if so,
			// no new lines to show. If it's a deletion within a line, the modified
			// line should still be shown.
			oldSpansFullLines := rep.OldLen > 0 && oldContent[rep.Offset+rep.OldLen-1] == '\n'
			if oldSpansFullLines {
				c.newEnd = c.newStart - 1 // Empty range — no new lines to show.
			} else {
				// Deletion within a line: the shortened line still exists.
				c.newEnd = c.newStart
				c.inserts = true
			}
		}

		changes = append(changes, c)
		shift += rep.NewLen - rep.OldLen
	}

	return changes
}

// groupChanges groups changes into hunk groups, merging those whose context lines would overlap.
func groupChanges(changes []change, contextLines, oldTotal, newTotal int) [][]change {
	if len(changes) == 0 {
		return nil
	}

	groups := [][]change{{changes[0]}}

	for _, c := range changes[1:] {
		lastGroup := groups[len(groups)-1]
		lastChange := lastGroup[len(lastGroup)-1]

		// Merge if the gap in old lines between end of last change and start of this
		// change is small enough that context windows would overlap.
		if c.oldStart-lastChange.oldEnd <= 2*contextLines+1 {
			groups[len(groups)-1] = append(groups[len(groups)-1], c)
		} else {
			groups = append(groups, []change{c})
		}
	}

	return groups
}

// buildHunk builds a single unified diff hunk from a group of changes.
func buildHunk(oldLines, newLines []string, changes []change, contextLines int) hunk {
	first := changes[0]
	last := changes[len(changes)-1]

	// Context-expanded boundaries (0-indexed).
	ctxOldStart := max(0, first.oldStart-contextLines)
	ctxOldEnd := min(len(oldLines)-1, last.oldEnd+contextLines)
	ctxNewStart := max(0, first.newStart-contextLines)
	var lines []string

	// Walk through old lines in the context-expanded range, inserting
	// context, removed, and added lines at the right positions.
	oldIdx := ctxOldStart

	for _, c := range changes {
		// Context lines before this change.
		for oldIdx < c.oldStart {
			lines = append(lines, " "+oldLines[oldIdx])
			oldIdx++
		}

		// Removed lines (old side).
		for i := c.oldStart; i <= c.oldEnd && i < len(oldLines); i++ {
			lines = append(lines, "-"+oldLines[i])
		}

		// Added lines (new side), only if the replacement produces new content.
		if c.inserts {
			for i := c.newStart; i <= c.newEnd && i < len(newLines); i++ {
				lines = append(lines, "+"+newLines[i])
			}
		}

		oldIdx = c.oldEnd + 1
	}

	// Trailing context.
	for oldIdx <= ctxOldEnd && oldIdx < len(oldLines) {
		lines = append(lines, " "+oldLines[oldIdx])
		oldIdx++
	}

	// Compute counts from the lines we produced.
	oldCount := 0
	newCount := 0
	for _, l := range lines {
		switch {
		case strings.HasPrefix(l, " "):
			oldCount++
			newCount++
		case strings.HasPrefix(l, "-"):
			oldCount++
		case strings.HasPrefix(l, "+"):
			newCount++
		}
	}

	return hunk{
		oldStart: ctxOldStart + 1,
		oldCount: oldCount,
		newStart: ctxNewStart + 1,
		newCount: newCount,
		lines:    lines,
	}
}

// ByteOffsetToLine returns the 0-indexed line number for a byte offset in content.
func ByteOffsetToLine(content string, offset int) int {
	if offset < 0 {
		return 0
	}
	if offset >= len(content) {
		offset = len(content) - 1
	}
	if offset < 0 {
		return 0
	}

	line := 0
	for i := range offset {
		if content[i] == '\n' {
			line++
		}
	}

	return line
}

// SplitLines splits content into lines. Unlike strings.Split, an empty string
// returns an empty slice and a trailing newline does not produce an extra empty element.
func SplitLines(s string) []string {
	if s == "" {
		return nil
	}

	lines := strings.Split(s, "\n")

	// Remove trailing empty element from final newline.
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	return lines
}
