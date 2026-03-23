package session_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/jsonl"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/edit"
	"github.com/slok/gosimov/pkg/tool/ls"
	"github.com/slok/gosimov/pkg/tool/read"
	"github.com/slok/gosimov/pkg/tool/write"
)

// newFileTools creates ls, read, write, and edit tools rooted at the given CWD.
func newFileTools(t *testing.T, cwd string) []tool.Tool {
	t.Helper()

	lsTool, err := ls.New(ls.Config{CWD: cwd})
	require.NoError(t, err)

	readTool, err := read.New(read.Config{CWD: cwd})
	require.NoError(t, err)

	writeTool, err := write.New(write.Config{CWD: cwd})
	require.NoError(t, err)

	editTool, err := edit.New(edit.Config{CWD: cwd})
	require.NoError(t, err)

	return []tool.Tool{lsTool, readTool, writeTool, editTool}
}

func TestToolUsageWriteEditRead(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	// Setup working directory for tools.
	cwd := t.TempDir()
	tools := newFileTools(t, cwd)

	// Setup store.
	repo, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	// Create provider with tools.
	provider := cfg.NewProvider(t)

	// Create session.
	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      "You are a helpful assistant. Use the available tools to complete tasks. Be concise. Do not ask for confirmation, just do it.",
		Tools:             tools,
		TurnMaxIterations: 15,
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	// Prompt: multi-step file task requiring write, edit, and read tools.
	result, err := promptWithRetry(t, ctx, session, []model.ContentPart{model.NewContentText(
		`Do the following steps using tools:
1. Write a file called "greeting.txt" with exactly this content (two lines):
hello world
goodbye world
2. Edit the file to replace "world" with "gosimov" on the FIRST line only (use old_text="hello world" and new_text="hello gosimov").
3. Read the file and tell me its exact contents.`)})
	require.NoError(t, err)

	// Verify the file was created with the expected final content.
	content, err := os.ReadFile(filepath.Join(cwd, "greeting.txt"))
	require.NoError(t, err, "greeting.txt should exist")
	fileContent := string(content)
	assert.Contains(t, fileContent, "hello gosimov", "first line should have been edited")
	assert.Contains(t, fileContent, "goodbye world", "second line should be unchanged")
	assert.NotContains(t, fileContent, "hello world", "original first line should be replaced")

	// Verify the agent loop used tools (messages should contain tool results).
	msgs := session.Messages()
	var toolResultCount int
	for _, m := range msgs {
		if m.Kind == model.MessageKindToolResult {
			toolResultCount++
		}
	}
	assert.GreaterOrEqual(t, toolResultCount, 3, "should have at least 3 tool results (write, edit, read)")

	// Verify LLM made multiple iterations (more than just user + single LLM response).
	assert.Greater(t, len(result.NewMessages), 1, "turn should have multiple messages (LLM + tool results)")

	// Token usage should be tracked across all iterations.
	assert.Greater(t, usageFromTurnMessages(result.NewMessages).TotalTokens, 0)
	assert.Greater(t, session.Usage().TotalTokens, 0)
}

func TestToolUsageListDirectory(t *testing.T) {
	cfg := NewConfig(t)

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	// Setup working directory with pre-existing files.
	cwd := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(cwd, "alpha.txt"), []byte("a"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(cwd, "beta.txt"), []byte("b"), 0644))
	require.NoError(t, os.Mkdir(filepath.Join(cwd, "subdir"), 0755))

	tools := newFileTools(t, cwd)

	// Setup store.
	repo, err := jsonl.New(jsonl.Config{Dir: t.TempDir()})
	require.NoError(t, err)

	provider := cfg.NewProvider(t)

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      "You are a helpful assistant. Use the available tools. Be concise.",
		Tools:             tools,
		TurnMaxIterations: 10,
		SessionRepository: repo,
		MessageRepository: repo,
	})
	require.NoError(t, err)

	// Prompt: list directory and report contents.
	result, err := promptWithRetry(t, ctx, session, []model.ContentPart{model.NewContentText(
		"List the files in the current directory using the ls tool and tell me what files exist.")})
	require.NoError(t, err)

	// The final response should mention the files.
	responseText := ""
	for _, cp := range finalLLMMessageFromTurnResult(t, result).Content {
		responseText += cp.Text
	}
	assert.Contains(t, responseText, "alpha.txt")
	assert.Contains(t, responseText, "beta.txt")

	// Should have used tools.
	var hasToolResult bool
	for _, m := range session.Messages() {
		if m.Kind == model.MessageKindToolResult {
			hasToolResult = true
			break
		}
	}
	assert.True(t, hasToolResult, "should have at least one tool result from ls")

	assert.Greater(t, usageFromTurnMessages(result.NewMessages).TotalTokens, 0)
}
