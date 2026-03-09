package jsonl

import (
	"encoding/json"
	"time"

	usageutil "github.com/slok/gosimov/internal/utils/usage"
	"github.com/slok/gosimov/pkg/model"
)

// JSONL line types — internal serialization format.
// Each line in a .jsonl session file has a "type" field that determines its schema.

// lineType identifies what kind of JSONL line this is.
type lineType string

const (
	lineTypeSession lineType = "session"
	lineTypeMessage lineType = "message"
)

// lineHeader is used for initial unmarshaling to determine the line type.
type lineHeader struct {
	Type lineType `json:"type"`
}

// sessionLine is the first line of a session file.
type sessionLine struct {
	Type      lineType  `json:"type"`
	ID        string    `json:"id"`
	CreatedAt time.Time `json:"created_at"`
}

// messageLine is a message entry in a session file.
type messageLine struct {
	Type             lineType            `json:"type"`
	ID               string              `json:"id"`
	Kind             model.MessageKind   `json:"kind"`
	Content          []contentPartLine   `json:"content,omitempty"`
	ToolCallRequests []toolCallLine      `json:"tool_call_requests,omitempty"`
	ToolCallID       string              `json:"tool_call_id,omitempty"`
	IsError          bool                `json:"is_error,omitempty"`
	Metadata         *metadataLine       `json:"metadata,omitempty"`
	Compaction       *compactionDataLine `json:"compaction,omitempty"`
	CreatedAt        time.Time           `json:"created_at"`
}

// contentPartLine is one piece of content in a message.
type contentPartLine struct {
	Type  model.ContentPartType `json:"type"`
	Text  string                `json:"text,omitempty"`
	Image *imageDataLine        `json:"image,omitempty"`
}

// imageDataLine holds image binary data (base64-encoded by json.Marshal).
type imageDataLine struct {
	Data     []byte `json:"data"`
	MimeType string `json:"mime_type"`
}

// toolCallLine is a tool call request from the LLM.
type toolCallLine struct {
	ID        string          `json:"id"`
	ToolID    string          `json:"tool_id"`
	Arguments json.RawMessage `json:"arguments"`
}

// metadataLine holds LLM response metadata.
type metadataLine struct {
	Usage      *usageLine       `json:"usage,omitempty"`
	StopReason model.StopReason `json:"stop_reason,omitempty"`
	Model      string           `json:"model,omitempty"`
	Provider   string           `json:"provider,omitempty"`
}

// usageLine holds token usage data.
type usageLine struct {
	InputTokens      int `json:"input_tokens,omitempty"`
	OutputTokens     int `json:"output_tokens,omitempty"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
}

// compactionDataLine holds compaction checkpoint data.
type compactionDataLine struct {
	FirstKeptID  string `json:"first_kept_id"`
	TokensBefore int    `json:"tokens_before,omitempty"`
}

// --- Session conversion ---

// sessionToLine converts a model.Session to a sessionLine.
func sessionToLine(s model.Session) sessionLine {
	return sessionLine{
		Type:      lineTypeSession,
		ID:        s.ID,
		CreatedAt: s.CreatedAt,
	}
}

// lineToSession converts a sessionLine to a model.Session.
func lineToSession(l sessionLine) model.Session {
	return model.Session{
		ID:        l.ID,
		CreatedAt: l.CreatedAt,
	}
}

// --- Message conversion ---

// messageToLine converts a model.Message to a messageLine.
func messageToLine(m model.Message) messageLine {
	ml := messageLine{
		Type:       lineTypeMessage,
		ID:         m.ID,
		Kind:       m.Kind,
		ToolCallID: m.ToolCallID,
		IsError:    m.IsError,
		CreatedAt:  m.CreatedAt,
	}

	// Content parts.
	if len(m.Content) > 0 {
		ml.Content = make([]contentPartLine, len(m.Content))
		for i, cp := range m.Content {
			ml.Content[i] = contentPartToLine(cp)
		}
	}

	// Tool call requests.
	if len(m.ToolCallRequests) > 0 {
		ml.ToolCallRequests = make([]toolCallLine, len(m.ToolCallRequests))
		for i, tc := range m.ToolCallRequests {
			ml.ToolCallRequests[i] = toolCallLine{
				ID:        tc.ID,
				ToolID:    tc.ToolID,
				Arguments: tc.Arguments,
			}
		}
	}

	// Metadata.
	if m.Metadata != nil {
		ml.Metadata = metadataToLine(m.Metadata)
	}

	// Compaction data.
	if m.Compaction != nil {
		ml.Compaction = &compactionDataLine{
			FirstKeptID:  m.Compaction.FirstKeptID,
			TokensBefore: m.Compaction.TokensBefore,
		}
	}

	return ml
}

// lineToMessage converts a messageLine to a model.Message.
func lineToMessage(l messageLine) model.Message {
	m := model.Message{
		ID:         l.ID,
		Kind:       l.Kind,
		ToolCallID: l.ToolCallID,
		IsError:    l.IsError,
		CreatedAt:  l.CreatedAt,
	}

	// Content parts.
	if len(l.Content) > 0 {
		m.Content = make([]model.ContentPart, len(l.Content))
		for i, cp := range l.Content {
			m.Content[i] = lineToContentPart(cp)
		}
	}

	// Tool call requests.
	if len(l.ToolCallRequests) > 0 {
		m.ToolCallRequests = make([]model.ToolCallRequest, len(l.ToolCallRequests))
		for i, tc := range l.ToolCallRequests {
			m.ToolCallRequests[i] = model.ToolCallRequest{
				ID:        tc.ID,
				ToolID:    tc.ToolID,
				Arguments: tc.Arguments,
			}
		}
	}

	// Metadata.
	if l.Metadata != nil {
		m.Metadata = lineToMetadata(l.Metadata)
	}

	// Compaction data.
	if l.Compaction != nil {
		m.Compaction = &model.CompactionData{
			FirstKeptID:  l.Compaction.FirstKeptID,
			TokensBefore: l.Compaction.TokensBefore,
		}
	}

	return m
}

// --- Content part conversion ---

func contentPartToLine(cp model.ContentPart) contentPartLine {
	cl := contentPartLine{Type: cp.Type}

	switch cp.Type {
	case model.ContentPartTypeText:
		cl.Text = cp.Text
	case model.ContentPartTypeImage:
		if cp.Image != nil {
			cl.Image = &imageDataLine{
				Data:     cp.Image.Data,
				MimeType: cp.Image.MimeType,
			}
		}
	}

	return cl
}

func lineToContentPart(cl contentPartLine) model.ContentPart {
	cp := model.ContentPart{Type: cl.Type}

	switch cl.Type {
	case model.ContentPartTypeText:
		cp.Text = cl.Text
	case model.ContentPartTypeImage:
		if cl.Image != nil {
			cp.Image = &model.ImageData{
				Data:     cl.Image.Data,
				MimeType: cl.Image.MimeType,
			}
		}
	}

	return cp
}

// --- Metadata conversion ---

func metadataToLine(md *model.MessageMetadata) *metadataLine {
	ml := &metadataLine{
		StopReason: md.StopReason,
		Model:      md.Model,
		Provider:   md.Provider,
	}

	if md.Usage != nil {
		u := usageutil.WithTotal(*md.Usage)
		ml.Usage = &usageLine{
			InputTokens:      u.InputTokens,
			OutputTokens:     u.OutputTokens,
			CacheReadTokens:  u.CacheReadTokens,
			CacheWriteTokens: u.CacheWriteTokens,
			TotalTokens:      u.TotalTokens,
			ReasoningTokens:  u.ReasoningTokens,
		}
	}

	return ml
}

func lineToMetadata(ml *metadataLine) *model.MessageMetadata {
	md := &model.MessageMetadata{
		StopReason: ml.StopReason,
		Model:      ml.Model,
		Provider:   ml.Provider,
	}

	if ml.Usage != nil {
		u := usageutil.WithTotal(model.Usage{
			InputTokens:      ml.Usage.InputTokens,
			OutputTokens:     ml.Usage.OutputTokens,
			CacheReadTokens:  ml.Usage.CacheReadTokens,
			CacheWriteTokens: ml.Usage.CacheWriteTokens,
			TotalTokens:      ml.Usage.TotalTokens,
			ReasoningTokens:  ml.Usage.ReasoningTokens,
		})

		md.Usage = &u
	}

	return md
}
