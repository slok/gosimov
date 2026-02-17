package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	usageutil "github.com/slok/gosimov/internal/utils/usage"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
)

type anthropicRequest struct {
	Model     string             `json:"model"`
	System    any                `json:"system,omitempty"`
	Messages  []anthropicMessage `json:"messages"`
	Tools     []anthropicTool    `json:"tools,omitempty"`
	MaxTokens int                `json:"max_tokens"`
}

type anthropicCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type anthropicTextBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicImageBlock struct {
	Type   string               `json:"type"`
	Source anthropicImageSource `json:"source"`
}

type anthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type anthropicToolUseBlock struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type anthropicToolResultBlock struct {
	Type         string                 `json:"type"`
	ToolUseID    string                 `json:"tool_use_id"`
	Content      string                 `json:"content"`
	IsError      bool                   `json:"is_error,omitempty"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicResponse struct {
	Model      string                  `json:"model"`
	StopReason string                  `json:"stop_reason"`
	Content    []anthropicResponsePart `json:"content"`
	Usage      *anthropicUsage         `json:"usage,omitempty"`
}

type anthropicResponsePart struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicUsage struct {
	InputTokens       int `json:"input_tokens"`
	OutputTokens      int `json:"output_tokens"`
	CacheReadTokens   int `json:"cache_read_input_tokens,omitempty"`
	CacheCreateTokens int `json:"cache_creation_input_tokens,omitempty"`
}

func convertTools(tools []tool.Tool, normalizeName func(string) string) []anthropicTool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]anthropicTool, len(tools))
	for i, t := range tools {
		name := t.ID()
		if normalizeName != nil {
			name = normalizeName(name)
		}

		result[i] = anthropicTool{
			Name:        name,
			Description: t.Description(),
			InputSchema: t.Schema(),
		}
	}

	return result
}

func convertMessages(messages []model.Message, normalizeToolName func(string) string) []anthropicMessage {
	result := make([]anthropicMessage, 0, len(messages))

	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		switch msg.Kind {
		case model.MessageKindUser:
			if converted, ok := convertUserMessage(msg); ok {
				result = append(result, converted)
			}
		case model.MessageKindLLM:
			if converted, ok := convertAssistantMessage(msg, normalizeToolName); ok {
				result = append(result, converted)
			}
		case model.MessageKindToolResult:
			converted, next := convertToolResultMessages(messages, i)
			if len(converted.Content.([]anthropicToolResultBlock)) > 0 {
				result = append(result, converted)
			}
			i = next - 1
		}
	}

	return result
}

func convertUserMessage(msg model.Message) (anthropicMessage, bool) {
	blocks := convertContentParts(msg.Content)
	if len(blocks) == 0 {
		return anthropicMessage{}, false
	}

	if len(blocks) == 1 {
		if tb, ok := blocks[0].(anthropicTextBlock); ok {
			return anthropicMessage{Role: "user", Content: tb.Text}, true
		}
	}

	return anthropicMessage{Role: "user", Content: blocks}, true
}

func convertAssistantMessage(msg model.Message, normalizeToolName func(string) string) (anthropicMessage, bool) {
	blocks := make([]any, 0, len(msg.Content)+len(msg.ToolCallRequests))

	for _, p := range msg.Content {
		if p.Type == model.ContentPartTypeText && strings.TrimSpace(p.Text) != "" {
			blocks = append(blocks, anthropicTextBlock{Type: "text", Text: p.Text})
		}
	}

	for _, tc := range msg.ToolCallRequests {
		name := tc.ToolID
		if normalizeToolName != nil {
			name = normalizeToolName(name)
		}

		args := tc.Arguments
		if len(strings.TrimSpace(string(args))) == 0 {
			args = json.RawMessage(`{}`)
		}

		blocks = append(blocks, anthropicToolUseBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  name,
			Input: args,
		})
	}

	if len(blocks) == 0 {
		return anthropicMessage{}, false
	}

	return anthropicMessage{Role: "assistant", Content: blocks}, true
}

func convertToolResultMessages(messages []model.Message, start int) (anthropicMessage, int) {
	blocks := make([]anthropicToolResultBlock, 0, 1)
	i := start

	for ; i < len(messages); i++ {
		msg := messages[i]
		if msg.Kind != model.MessageKindToolResult {
			break
		}

		if strings.TrimSpace(msg.ToolCallID) == "" {
			continue
		}

		content := extractText(msg.Content)
		if content == "" {
			content = "(no output)"
		}

		blocks = append(blocks, anthropicToolResultBlock{
			Type:      "tool_result",
			ToolUseID: msg.ToolCallID,
			Content:   content,
			IsError:   msg.IsError,
		})
	}

	return anthropicMessage{Role: "user", Content: blocks}, i
}

func convertContentParts(parts []model.ContentPart) []any {
	result := make([]any, 0, len(parts))

	for _, p := range parts {
		switch p.Type {
		case model.ContentPartTypeText:
			if strings.TrimSpace(p.Text) != "" {
				result = append(result, anthropicTextBlock{Type: "text", Text: p.Text})
			}
		case model.ContentPartTypeImage:
			if p.Image == nil || len(p.Image.Data) == 0 {
				continue
			}

			result = append(result, anthropicImageBlock{
				Type: "image",
				Source: anthropicImageSource{
					Type:      "base64",
					MediaType: p.Image.MimeType,
					Data:      base64.StdEncoding.EncodeToString(p.Image.Data),
				},
			})
		}
	}

	return result
}

func convertResponse(resp anthropicResponse, restoreToolName func(string) string) model.Message {
	msg := model.Message{
		Kind: model.MessageKindLLM,
		Metadata: &model.MessageMetadata{
			Model:      resp.Model,
			Usage:      convertUsage(resp.Usage),
			StopReason: convertStopReason(resp.StopReason),
		},
	}

	for _, p := range resp.Content {
		switch p.Type {
		case "text":
			if strings.TrimSpace(p.Text) != "" {
				msg.Content = append(msg.Content, model.ContentPart{Type: model.ContentPartTypeText, Text: p.Text})
			}
		case "tool_use":
			if strings.TrimSpace(p.ID) == "" || strings.TrimSpace(p.Name) == "" {
				continue
			}

			name := p.Name
			if restoreToolName != nil {
				name = restoreToolName(name)
			}

			args := p.Input
			if len(strings.TrimSpace(string(args))) == 0 {
				args = json.RawMessage(`{}`)
			}

			msg.ToolCallRequests = append(msg.ToolCallRequests, model.ToolCallRequest{
				ID:        p.ID,
				ToolID:    name,
				Arguments: args,
			})
		}
	}

	if len(msg.ToolCallRequests) > 0 {
		msg.Metadata.StopReason = model.StopReasonToolUse
	}

	return msg
}

func convertStopReason(reason string) model.StopReason {
	switch reason {
	case "end_turn", "stop_sequence", "pause_turn":
		return model.StopReasonComplete
	case "max_tokens":
		return model.StopReasonMaxTokens
	case "tool_use":
		return model.StopReasonToolUse
	case "refusal", "sensitive":
		return model.StopReasonError
	default:
		return model.StopReasonComplete
	}
}

func convertUsage(u *anthropicUsage) *model.Usage {
	if u == nil {
		return nil
	}

	normalized := usageutil.Normalize(model.Usage{
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens,
		CacheWriteTokens: u.CacheCreateTokens,
	}, false)

	return &normalized
}

func extractText(parts []model.ContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Type == model.ContentPartTypeText && strings.TrimSpace(p.Text) != "" {
			texts = append(texts, p.Text)
		}
	}

	return strings.Join(texts, "\n")
}

type apiErrorResponse struct {
	Type  string         `json:"type"`
	Error apiErrorDetail `json:"error"`
}

type apiErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func parseAPIError(statusCode int, body []byte) error {
	var errResp apiErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil {
		return fmt.Errorf("api error (status %d): %s", statusCode, strings.TrimSpace(string(body)))
	}

	msg := strings.TrimSpace(errResp.Error.Message)
	if msg == "" {
		msg = strings.TrimSpace(string(body))
	}
	if msg == "" {
		msg = "unknown api error"
	}

	errType := strings.TrimSpace(errResp.Error.Type)
	if errType == "" {
		errType = strings.TrimSpace(errResp.Type)
	}

	return fmt.Errorf("api error (status %d, type=%s): %s", statusCode, errType, msg)
}
