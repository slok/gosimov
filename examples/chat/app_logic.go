package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/edit"
	"github.com/slok/gosimov/pkg/tool/ls"
	"github.com/slok/gosimov/pkg/tool/read"
	"github.com/slok/gosimov/pkg/tool/shell"
	"github.com/slok/gosimov/pkg/tool/write"
)

func loadAllMessages(ctx context.Context, repo store.MessageRepository, sessionID string) ([]model.Message, error) {
	all := []model.Message{}
	opts := store.ListOpts{Limit: 200}

	for {
		result, err := repo.ListMessages(ctx, sessionID, opts)
		if err != nil {
			return nil, err
		}

		all = append(all, result.Items...)
		if result.NextCursor == "" {
			return all, nil
		}

		opts.Cursor = result.NextCursor
	}
}

// trimLoadedMessages keeps a bounded tail of loaded history and then sanitizes
// tool-use/tool-result structure so resumed sessions are valid for strict
// providers (Anthropic requires each tool_use to be immediately followed by
// matching tool_result blocks).
func trimLoadedMessages(messages []model.Message, max int) []model.Message {
	if max <= 0 || len(messages) <= max {
		trimmed := make([]model.Message, len(messages))
		copy(trimmed, messages)

		return sanitizeLoadedMessages(trimmed)
	}

	start := len(messages) - max
	for start < len(messages) && messages[start].Kind == model.MessageKindToolResult {
		start++
	}

	trimmed := make([]model.Message, len(messages)-start)
	copy(trimmed, messages[start:])

	return sanitizeLoadedMessages(trimmed)
}

func sanitizeLoadedMessages(messages []model.Message) []model.Message {
	// Rebuild history as a sequence of:
	//   - regular non-tool messages, and
	//   - complete tool-use blocks (LLM tool_use + immediate matching tool_results).
	//
	// Why: persisted sessions can contain interrupted tails (e.g. after crashes or
	// cancellation) where tool_use blocks are incomplete. Sending those back to
	// Anthropic yields a 400 because tool_result blocks must immediately follow the
	// corresponding tool_use.
	if len(messages) == 0 {
		return nil
	}

	out := make([]model.Message, 0, len(messages))
	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		if msg.Kind == model.MessageKindToolResult {
			// Standalone tool_result is always invalid without a preceding tool_use
			// block in this sanitized stream, so drop it.
			continue
		}

		if msg.Kind == model.MessageKindLLM && len(msg.ToolCallRequests) > 0 {
			// Collect the immediate tool_result run after this tool_use message.
			end := i + 1
			for end < len(messages) && messages[end].Kind == model.MessageKindToolResult {
				end++
			}

			// Keep this block only when it is complete and consistent.
			if hasValidToolResultBlock(msg, messages[i+1:end]) {
				out = append(out, msg)
				out = append(out, messages[i+1:end]...)
			}

			// Skip over the consumed tool_result run regardless of validity.
			i = end - 1
			continue
		}

		out = append(out, msg)
	}

	return out
}

func hasValidToolResultBlock(toolUseMsg model.Message, toolResults []model.Message) bool {
	// A valid block requires:
	//   - at least one requested tool call,
	//   - at least one immediate tool_result,
	//   - every result refers to one requested call,
	//   - every requested call has a corresponding result.
	//
	// This guarantees we only keep sequences that are accepted when replayed to
	// providers that enforce strict tool_use/tool_result adjacency.
	if len(toolUseMsg.ToolCallRequests) == 0 || len(toolResults) == 0 {
		return false
	}

	required := make(map[string]struct{}, len(toolUseMsg.ToolCallRequests))
	for _, req := range toolUseMsg.ToolCallRequests {
		if req.ID == "" {
			return false
		}
		required[req.ID] = struct{}{}
	}

	seen := make(map[string]struct{}, len(toolResults))
	for _, tr := range toolResults {
		if tr.ToolCallID == "" {
			return false
		}
		if _, ok := required[tr.ToolCallID]; !ok {
			return false
		}
		seen[tr.ToolCallID] = struct{}{}
	}

	for id := range required {
		if _, ok := seen[id]; !ok {
			return false
		}
	}

	return true
}

func toExportMessages(msgs []model.Message) []exportMessage {
	result := make([]exportMessage, 0, len(msgs))
	for _, msg := range msgs {
		em := exportMessage{
			Kind:      string(msg.Kind),
			CreatedAt: msg.CreatedAt,
			IsError:   msg.IsError,
		}

		if msg.ToolCallID != "" {
			em.ToolCallID = msg.ToolCallID
		}

		for _, part := range msg.Content {
			switch part.Type {
			case model.ContentPartTypeText:
				if part.Text != "" {
					if em.Text != "" {
						em.Text += "\n"
					}
					em.Text += part.Text
				}
			case model.ContentPartTypeImage:
				if em.Text != "" {
					em.Text += "\n"
				}
				em.Text += "[image]"
			}
		}

		for _, tc := range msg.ToolCallRequests {
			em.ToolCalls = append(em.ToolCalls, fmt.Sprintf("%s(%s)", tc.ToolID, string(tc.Arguments)))
		}

		result = append(result, em)
	}

	return result
}

func (a *app) getSession(id string) *chatSession {
	a.mu.RLock()
	defer a.mu.RUnlock()

	return a.sessions[id]
}

func createToolsForDir(workDir string) ([]tool.Tool, error) {
	lsTool, err := ls.New(ls.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating ls tool: %w", err)
	}

	readTool, err := read.New(read.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating read tool: %w", err)
	}

	writeTool, err := write.New(write.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating write tool: %w", err)
	}

	editTool, err := edit.New(edit.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating edit tool: %w", err)
	}

	shellTool, err := shell.New(shell.Config{CWD: workDir})
	if err != nil {
		return nil, fmt.Errorf("creating shell tool: %w", err)
	}

	tools := []tool.Tool{lsTool, readTool, writeTool, editTool, shellTool}

	return tools, nil
}

func defaultSystemPrompt(v string) string {
	v = strings.TrimSpace(v)
	if v != "" {
		return v
	}

	return "You are a concise coding assistant. Use tools autonomously when helpful. Explain what you changed and why."
}
