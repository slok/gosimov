package openaichat

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	usageutil "github.com/slok/gosimov/internal/utils/usage"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
)

type chatRequest struct {
	Model          string        `json:"model"`
	Messages       []chatMessage `json:"messages"`
	PromptCacheKey string        `json:"prompt_cache_key,omitempty"`
	Tools          []chatTool    `json:"tools,omitempty"`
	MaxTokens      int           `json:"max_tokens,omitempty"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolCalls  []chatToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type chatContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
}

type chatImageURL struct {
	URL string `json:"url"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage,omitempty"`
	Model   string       `json:"model"`
}

type chatChoice struct {
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int              `json:"prompt_tokens"`
	CompletionTokens int              `json:"completion_tokens"`
	PromptDetails    *chatUsageDetail `json:"prompt_tokens_details,omitempty"`
}

type chatUsageDetail struct {
	CachedTokens int `json:"cached_tokens"`
}

func convertTools(tools []tool.Tool) []chatTool {
	if len(tools) == 0 {
		return nil
	}
	result := make([]chatTool, len(tools))
	for i, t := range tools {
		result[i] = chatTool{Type: "function", Function: chatFunction{Name: t.ID(), Description: t.Description(), Parameters: t.Schema()}}
	}
	return result
}

func convertMessages(systemPrompt string, messages []model.Message) []chatMessage {
	var result []chatMessage
	if systemPrompt != "" {
		content, _ := json.Marshal(systemPrompt)
		result = append(result, chatMessage{Role: "system", Content: content})
	}
	for _, msg := range messages {
		if converted := convertMessage(msg); converted != nil {
			result = append(result, *converted)
		}
	}
	return result
}

func convertMessage(msg model.Message) *chatMessage {
	switch msg.Kind {
	case model.MessageKindUser:
		return convertUserMessage(msg)
	case model.MessageKindLLM:
		return convertLLMMessage(msg)
	case model.MessageKindToolResult:
		return convertToolResultMessage(msg)
	default:
		return nil
	}
}

func convertUserMessage(msg model.Message) *chatMessage {
	parts := convertContentParts(msg.Content)
	if len(parts) == 0 {
		return nil
	}
	if len(parts) == 1 && parts[0].Type == "text" {
		content, _ := json.Marshal(parts[0].Text)
		return &chatMessage{Role: "user", Content: content}
	}
	content, _ := json.Marshal(parts)
	return &chatMessage{Role: "user", Content: content}
}

func convertLLMMessage(msg model.Message) *chatMessage {
	cm := &chatMessage{Role: "assistant"}
	text := extractText(msg.Content)
	if text != "" {
		content, _ := json.Marshal(text)
		cm.Content = content
	}
	if len(msg.ToolCallRequests) > 0 {
		cm.ToolCalls = make([]chatToolCall, len(msg.ToolCallRequests))
		for i, tc := range msg.ToolCallRequests {
			cm.ToolCalls[i] = chatToolCall{ID: tc.ID, Type: "function", Function: chatFunctionCall{Name: tc.ToolID, Arguments: string(tc.Arguments)}}
		}
	}
	return cm
}

func convertToolResultMessage(msg model.Message) *chatMessage {
	if strings.TrimSpace(msg.ToolCallID) == "" {
		return nil
	}
	content, _ := json.Marshal(extractText(msg.Content))
	return &chatMessage{Role: "tool", ToolCallID: msg.ToolCallID, Content: content}
}

func convertContentParts(parts []model.ContentPart) []chatContentPart {
	var result []chatContentPart
	for _, p := range parts {
		switch p.Type {
		case model.ContentPartTypeText:
			if p.Text != "" {
				result = append(result, chatContentPart{Type: "text", Text: p.Text})
			}
		case model.ContentPartTypeImage:
			if p.Image != nil && len(p.Image.Data) > 0 {
				dataURI := fmt.Sprintf("data:%s;base64,%s", p.Image.MimeType, base64.StdEncoding.EncodeToString(p.Image.Data))
				result = append(result, chatContentPart{Type: "image_url", ImageURL: &chatImageURL{URL: dataURI}})
			}
		}
	}
	return result
}

func extractText(parts []model.ContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func convertStopReason(reason string) model.StopReason {
	switch reason {
	case "stop":
		return model.StopReasonComplete
	case "length":
		return model.StopReasonMaxTokens
	case "tool_calls":
		return model.StopReasonToolUse
	case "content_filter":
		return model.StopReasonError
	default:
		return model.StopReasonComplete
	}
}

func convertUsage(u *chatUsage) *model.Usage {
	if u == nil {
		return nil
	}

	raw := model.Usage{InputTokens: u.PromptTokens, OutputTokens: u.CompletionTokens}
	if u.PromptDetails != nil {
		raw.CacheReadTokens = u.PromptDetails.CachedTokens
	}

	normalized := usageutil.Normalize(raw, true)

	return &normalized
}

func convertResponse(resp chatResponse) model.Message {
	msg := model.Message{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{Model: resp.Model, Usage: convertUsage(resp.Usage)}}
	if len(resp.Choices) == 0 {
		msg.Metadata.StopReason = model.StopReasonComplete
		return msg
	}
	choice := resp.Choices[0]
	msg.Metadata.StopReason = convertStopReason(choice.FinishReason)
	var text string
	if choice.Message.Content != nil {
		_ = json.Unmarshal(choice.Message.Content, &text)
	}
	if text != "" {
		msg.Content = []model.ContentPart{model.NewContentText(text)}
	}
	if len(choice.Message.ToolCalls) > 0 {
		msg.ToolCallRequests = make([]model.ToolCallRequest, len(choice.Message.ToolCalls))
		for i, tc := range choice.Message.ToolCalls {
			msg.ToolCallRequests[i] = model.ToolCallRequest{ID: tc.ID, ToolID: tc.Function.Name, Arguments: json.RawMessage(tc.Function.Arguments)}
		}
		msg.Metadata.StopReason = model.StopReasonToolUse
	}
	return msg
}
