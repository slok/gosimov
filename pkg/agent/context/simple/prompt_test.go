package simple

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMakeSummarizationPrompt(t *testing.T) {
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
