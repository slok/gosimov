// Example simple demonstrates a multi-tool agent session using a fake LLM provider.
//
// The fake provider scripts a realistic coding workflow:
//  1. ls — check the working directory
//  2. write — create a Go hello world program
//  3. shell — run the program
//  4. read — read the file for context
//  5. edit — change the greeting
//  6. shell — run again to verify
//
// All 5 tools (ls, read, write, edit, shell) execute against real files in a temp directory.
//
// Usage:
//
//	go run ./examples/simple
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/fake"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/memory"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/edit"
	"github.com/slok/gosimov/pkg/tool/ls"
	"github.com/slok/gosimov/pkg/tool/read"
	"github.com/slok/gosimov/pkg/tool/shell"
	"github.com/slok/gosimov/pkg/tool/write"
)

const helloWorldProgram = `package main

import "fmt"

func main() {
	fmt.Println("Hello, World!")
}
`

// scriptedProvider creates a fake LLM that walks through a scripted coding workflow.
// It tracks which tool calls have been made by counting tool result messages.
func scriptedProvider() llm.Provider {
	return fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
		// Count tool results to determine which step we're on.
		toolResults := 0
		for _, msg := range req.Messages {
			if msg.Kind == model.MessageKindToolResult {
				toolResults++
			}
		}

		switch toolResults {
		case 0:
			// No tools called yet — check the directory.
			return toolCall("tc-1", "ls", `{"path": "."}`), nil

		case 1:
			// ls done — write main.go.
			return toolCall("tc-2", "write", fmt.Sprintf(`{"path": "main.go", "content": %s}`, mustJSON(helloWorldProgram))), nil

		case 2:
			// write done — run the program.
			return toolCall("tc-3", "shell", `{"command": "go run main.go"}`), nil

		case 3:
			// first run done — read the file for edit context.
			return toolCall("tc-4", "read", `{"path": "main.go"}`), nil

		case 4:
			// read done — edit the greeting.
			return toolCall("tc-5", "edit", `{"path": "main.go", "old_text": "Hello, World!", "new_text": "Hello, Gosimov!"}`), nil

		case 5:
			// edit done — run again to verify.
			return toolCall("tc-6", "shell", `{"command": "go run main.go"}`), nil

		default:
			// All done — summarize.
			return complete("Done! I created a Go hello world program, ran it, " +
				"changed the greeting to \"Hello, Gosimov!\", and verified it works."), nil
		}
	})
}

func run(ctx context.Context) error {
	// Create a temp directory as the working directory for all tools.
	workDir, err := os.MkdirTemp("", "gosimov-example-*")
	if err != nil {
		return fmt.Errorf("creating temp dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	fmt.Printf("Working directory: %s\n\n", workDir)

	// Initialize all tools pointing at the same working directory.
	lsTool, err := ls.New(ls.Config{CWD: workDir})
	if err != nil {
		return fmt.Errorf("creating ls tool: %w", err)
	}

	readTool, err := read.New(read.Config{CWD: workDir})
	if err != nil {
		return fmt.Errorf("creating read tool: %w", err)
	}

	writeTool, err := write.New(write.Config{CWD: workDir})
	if err != nil {
		return fmt.Errorf("creating write tool: %w", err)
	}

	editTool, err := edit.New(edit.Config{CWD: workDir})
	if err != nil {
		return fmt.Errorf("creating edit tool: %w", err)
	}

	shellTool, err := shell.New(shell.Config{CWD: workDir})
	if err != nil {
		return fmt.Errorf("creating shell tool: %w", err)
	}

	// Create the session.
	repo := memory.NewRepository()

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          scriptedProvider(),
		SystemPrompt:      "You are a helpful coding assistant.",
		Tools:             []tool.Tool{lsTool, readTool, writeTool, editTool, shellTool},
		SessionRepository: repo,
		MessageRepository: repo,
	})
	if err != nil {
		return fmt.Errorf("creating session: %w", err)
	}

	fmt.Printf("Session: %s\n\n", session.Session().ID)

	// Run the prompt.
	fmt.Println("User: Create a Go hello world, run it, change greeting to 'Hello, Gosimov!', run again.")
	fmt.Println()

	result, err := session.Prompt(ctx, []model.ContentPart{
		{Type: model.ContentPartTypeText, Text: "Create a Go hello world program, run it, then change the greeting to say 'Hello, Gosimov!' and run it again."},
	}, agent.PromptOptions{})
	if err != nil {
		return fmt.Errorf("prompt: %w", err)
	}

	// Print each message from the turn.
	fmt.Println("--- Turn messages ---")
	for _, msg := range result.Messages {
		printMessage(msg)
	}

	// Summary.
	fmt.Println("--- Summary ---")
	fmt.Printf("Messages in turn: %d\n", len(result.Messages))
	fmt.Printf("Total messages:   %d\n", len(session.Messages()))
	usage := session.Usage()
	fmt.Printf("Tokens:           %d total, %d input, %d output\n", usage.TotalTokens, usage.InputTokens, usage.OutputTokens)

	return nil
}

func printMessage(msg model.Message) {
	switch msg.Kind {
	case model.MessageKindLLM:
		if len(msg.ToolCallRequests) > 0 {
			ids := make([]string, 0, len(msg.ToolCallRequests))
			for _, tc := range msg.ToolCallRequests {
				ids = append(ids, tc.ToolID)
			}
			fmt.Printf("  LLM  -> tool call: %s\n", strings.Join(ids, ", "))
		} else if len(msg.Content) > 0 {
			text := msg.Content[0].Text
			if len(text) > 120 {
				text = text[:120] + "..."
			}
			fmt.Printf("  LLM  -> %s\n", text)
		}

	case model.MessageKindToolResult:
		text := "(no content)"
		if len(msg.Content) > 0 {
			text = msg.Content[0].Text
		}
		// Truncate for display.
		if len(text) > 100 {
			text = text[:100] + "..."
		}
		errTag := ""
		if msg.IsError {
			errTag = " [ERROR]"
		}
		fmt.Printf("  Tool -> %s%s\n", text, errTag)
	}
}

// toolCall creates an LLM response that requests a tool call.
func toolCall(id, toolID, argsJSON string) *llm.Response {
	return &llm.Response{
		Message: model.Message{
			Kind: model.MessageKindLLM,
			ToolCallRequests: []model.ToolCallRequest{
				{ID: id, ToolID: toolID, Arguments: json.RawMessage(argsJSON)},
			},
			Metadata: &model.MessageMetadata{
				StopReason: model.StopReasonToolUse,
				Usage:      &model.Usage{InputTokens: 50, OutputTokens: 25},
			},
		},
	}
}

// complete creates an LLM response that ends the turn.
func complete(text string) *llm.Response {
	return &llm.Response{
		Message: model.Message{
			Kind:    model.MessageKindLLM,
			Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: text}},
			Metadata: &model.MessageMetadata{
				StopReason: model.StopReasonComplete,
				Usage:      &model.Usage{InputTokens: 100, OutputTokens: 50},
			},
		},
	}
}

// mustJSON marshals a value to a JSON string.
func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}
