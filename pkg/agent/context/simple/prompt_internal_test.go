package simple

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMakeSummarizationPrompt(t *testing.T) {
	tests := map[string]struct {
		conversation       string
		previousSummary    string
		customInstructions string
		assert             func(t *testing.T, got string)
	}{
		"Without previous summary should use initial template.": {
			conversation: "[User]: hello",
			assert: func(t *testing.T, got string) {
				assert := assert.New(t)

				assert.Contains(got, "<conversation>\n[User]: hello\n</conversation>")
				assert.Contains(got, initialSummarizationPrompt)
				assert.NotContains(got, "<previous-summary>")
				assert.NotContains(got, updateSummarizationPrompt)
			},
		},
		"With previous summary should use update template and include previous section.": {
			conversation:    "[User]: hello",
			previousSummary: "## Goal\n- test",
			assert: func(t *testing.T, got string) {
				assert := assert.New(t)

				assert.Contains(got, "<conversation>\n[User]: hello\n</conversation>")
				assert.Contains(got, "<previous-summary>\n## Goal\n- test\n</previous-summary>")
				assert.Contains(got, updateSummarizationPrompt)
				assert.NotContains(got, initialSummarizationPrompt)
			},
		},
		"Custom instructions should be included.": {
			conversation:       "[User]: hello",
			customInstructions: "focus on auth",
			assert: func(t *testing.T, got string) {
				assert.Contains(t, got, "Custom instructions: focus on auth")
			},
		},
		"Sections should be separated by blank lines.": {
			conversation:       "[User]: hello",
			previousSummary:    "sum",
			customInstructions: "focus",
			assert: func(t *testing.T, got string) {
				assert.GreaterOrEqual(t, strings.Count(got, "\n\n"), 3)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := makeSummarizationPrompt(test.conversation, test.previousSummary, test.customInstructions)
			test.assert(t, got)
		})
	}
}

func TestMakeSummarizationPromptExact(t *testing.T) {
	tests := map[string]struct {
		conversation       string
		previousSummary    string
		customInstructions string
		exp                string
	}{
		"Initial prompt exact output.": {
			conversation: "[User]: hello",
			exp:          fmt.Sprintf("<conversation>\n%s\n</conversation>\n\n%s", "[User]: hello", initialSummarizationPrompt),
		},
		"Update prompt exact output with previous summary.": {
			conversation:    "[User]: hello",
			previousSummary: "## Goal\n- test",
			exp:             fmt.Sprintf("<conversation>\n%s\n</conversation>\n\n<previous-summary>\n%s\n</previous-summary>\n\n%s", "[User]: hello", "## Goal\n- test", updateSummarizationPrompt),
		},
		"Update prompt exact output with previous summary and custom instructions.": {
			conversation:       "[User]: hello",
			previousSummary:    "## Goal\n- test",
			customInstructions: "focus on auth",
			exp:                fmt.Sprintf("<conversation>\n%s\n</conversation>\n\n<previous-summary>\n%s\n</previous-summary>\n\nCustom instructions: %s\n\n%s", "[User]: hello", "## Goal\n- test", "focus on auth", updateSummarizationPrompt),
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := makeSummarizationPrompt(test.conversation, test.previousSummary, test.customInstructions)
			assert.Equal(t, test.exp, got)
		})
	}
}
