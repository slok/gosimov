package tool

import (
	"context"
	"encoding/json"

	"github.com/slok/gosimov/pkg/model"
)

// Tool is something the LLM can request to execute.
//
// The agent loop discovers available tools, sends their schemas to the LLM,
// and executes them when the LLM issues a [model.ToolCallRequest].
type Tool interface {
	// ID is the unique identifier for this tool (what the LLM references).
	ID() string
	// Description explains what the tool does (sent to the LLM).
	Description() string
	// Schema returns the JSON Schema for the tool's arguments.
	Schema() json.RawMessage
	// Execute runs the tool with the given arguments.
	Execute(ctx context.Context, args json.RawMessage) (*Result, error)
}

// Result is what a tool returns after successful execution.
//
// If a tool fails, it returns a Go error instead. The agent loop
// sends err.Error() back to the LLM as an error tool result.
type Result struct {
	Content []model.ContentPart
}
