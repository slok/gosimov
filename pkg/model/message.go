package model

import (
	"encoding/json"
	"time"
)

// MessageKind identifies what kind of message this is.
type MessageKind string

const (
	// MessageKindUser is a message from the user (human or system input).
	MessageKindUser MessageKind = "user"
	// MessageKindLLM is a message produced by the LLM.
	MessageKindLLM MessageKind = "llm"
	// MessageKindToolResult is the result of executing a tool call.
	MessageKindToolResult MessageKind = "tool_result"
	// MessageKindCompaction is a compaction checkpoint that summarizes older messages.
	// When present, messages before the checkpoint (identified by Compaction.FirstKeptID)
	// are replaced by the summary in Content for LLM context building.
	MessageKindCompaction MessageKind = "compaction"
)

// ContentPartType identifies what kind of displayable content a ContentPart holds.
type ContentPartType string

const (
	// ContentPartTypeText is plain text content.
	ContentPartTypeText ContentPartType = "text"
	// ContentPartTypeImage is image binary content.
	ContentPartTypeImage ContentPartType = "image"
)

// StopReason indicates why the LLM stopped generating.
//
// Each value represents a distinct end condition that callers can branch on,
// similar in spirit to sentinel errors.
type StopReason string

const (
	// StopReasonNone is the zero value — no stop reason has been set.
	// This is the default for in-progress or unfinished responses.
	StopReasonNone StopReason = ""
	// StopReasonComplete means the model finished its response naturally (nothing more to say).
	StopReasonComplete StopReason = "complete"
	// StopReasonMaxTokens means the model hit the max output token limit mid-response.
	// The response is truncated.
	StopReasonMaxTokens StopReason = "max_tokens"
	// StopReasonToolUse means the model wants to call one or more tools.
	// The agent loop should execute the tool calls and continue.
	StopReasonToolUse StopReason = "tool_use"
	// StopReasonError means the LLM call failed (API error, network, etc.).
	StopReasonError StopReason = "error"
	// StopReasonAborted means the request was cancelled (context cancelled, user abort, timeout).
	// Partial content may exist.
	StopReasonAborted StopReason = "aborted"
)

// Message is a single message in a conversation.
//
// Messages are the fundamental unit of storage — they are stored flat, not grouped
// into turns. Turn grouping is derived at runtime via [TurnsFromMessages].
//
// Fields are populated based on Kind:
//   - MessageKindUser: Content.
//   - MessageKindLLM: Content, ToolCallRequests, Metadata.
//   - MessageKindToolResult: Content, ToolCallID, IsError.
//   - MessageKindCompaction: Content (summary text), Compaction.
type Message struct {
	ID               string
	Kind             MessageKind
	Content          []ContentPart
	ToolCallRequests []ToolCallRequest // MessageKindLLM only: tool calls requested by the LLM.
	ToolCallID       string            // MessageKindToolResult only: references ToolCallRequest.ID.
	IsError          bool              // MessageKindToolResult only: true if the tool execution failed.
	Metadata         *MessageMetadata  // MessageKindLLM only: LLM response metadata.
	Compaction       *CompactionData   // MessageKindCompaction only: compaction checkpoint data.
	CreatedAt        time.Time
}

// CompactionData holds metadata for a compaction checkpoint message.
//
// A compaction summarizes older messages in the conversation. The summary text
// is stored in the message's Content field. CompactionData records which messages
// were compacted so the context builder knows where to resume.
type CompactionData struct {
	// FirstKeptID is the ID of the first message that was kept (not summarized).
	// Messages before this ID in the conversation are covered by the summary.
	// When building LLM context, everything before FirstKeptID is replaced by the summary.
	FirstKeptID string
	// TokensBefore is the estimated context size (in tokens) before compaction.
	// Used for analytics and debugging, not for logic.
	TokensBefore int
}

// MessageMetadata holds LLM-response metadata attached to LLM messages.
type MessageMetadata struct {
	Usage      *Usage
	StopReason StopReason
	Model      string // Model ID that produced this message.
	Provider   string // Provider ID (e.g., "opencode-go", "openai").
	// ProviderInternalData carries provider-specific metadata required to continue a conversation.
	ProviderInternalData map[string]string
}

// ContentPart is one piece of displayable content within a message.
//
// Only one of the typed fields is set, determined by Type.
type ContentPart struct {
	Type  ContentPartType
	Text  string     // Set when Type == ContentPartTypeText.
	Image *ImageData // Set when Type == ContentPartTypeImage.
}

// ToolCallRequest represents an LLM's request to invoke a tool.
//
// The agent loop receives these from LLM messages and decides whether
// to execute them (based on permissions, etc.).
type ToolCallRequest struct {
	ID        string          // Unique ID for this request (links to Message.ToolCallID on results).
	ToolID    string          // Identifies which tool to call.
	Arguments json.RawMessage // Parameters for the tool as JSON.
}

// ImageData holds image binary data.
type ImageData struct {
	Data     []byte // Raw image bytes.
	MimeType string // e.g., "image/png", "image/jpeg".
}
