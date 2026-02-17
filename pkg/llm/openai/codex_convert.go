package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	usageutil "github.com/slok/gosimov/internal/utils/usage"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
)

type codexRequest struct {
	Model             string      `json:"model"`
	Store             bool        `json:"store"`
	Stream            bool        `json:"stream"`
	Instructions      string      `json:"instructions,omitempty"`
	Input             []any       `json:"input"`
	PromptCacheKey    string      `json:"prompt_cache_key,omitempty"`
	PromptCacheTTL    string      `json:"prompt_cache_retention,omitempty"`
	Tools             []codexTool `json:"tools,omitempty"`
	Text              *codexText  `json:"text,omitempty"`
	Include           []string    `json:"include,omitempty"`
	ToolChoice        string      `json:"tool_choice,omitempty"`
	ParallelToolCalls bool        `json:"parallel_tool_calls,omitempty"`
	MaxOutputTokens   int         `json:"max_output_tokens,omitempty"`
}

type codexText struct {
	Verbosity string `json:"verbosity,omitempty"`
}

type codexTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict"`
}

type codexResponse struct {
	Model  string            `json:"model"`
	Status string            `json:"status"`
	Output []codexOutputItem `json:"output"`
	Usage  *codexUsage       `json:"usage,omitempty"`
}

type codexOutputItem struct {
	Type      string               `json:"type"`
	Role      string               `json:"role,omitempty"`
	ID        string               `json:"id,omitempty"`
	CallID    string               `json:"call_id,omitempty"`
	Name      string               `json:"name,omitempty"`
	Arguments string               `json:"arguments,omitempty"`
	Content   []codexResponseBlock `json:"content,omitempty"`
}

type codexResponseBlock struct {
	Type    string `json:"type"`
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

type codexUsage struct {
	InputTokens        int                    `json:"input_tokens"`
	OutputTokens       int                    `json:"output_tokens"`
	InputTokensDetails *codexInputTokenDetail `json:"input_tokens_details,omitempty"`
}

type codexInputTokenDetail struct {
	CachedTokens int `json:"cached_tokens"`
}

func convertCodexTools(tools []tool.Tool) []codexTool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]codexTool, len(tools))
	for i, t := range tools {
		result[i] = codexTool{
			Type:        "function",
			Name:        t.ID(),
			Description: t.Description(),
			Parameters:  t.Schema(),
			Strict:      nil,
		}
	}

	return result
}

func convertCodexMessages(messages []model.Message) []any {
	result := make([]any, 0, len(messages))

	for _, msg := range messages {
		switch msg.Kind {
		case model.MessageKindUser:
			if m, ok := convertCodexUserMessage(msg); ok {
				result = append(result, m)
			}
		case model.MessageKindLLM:
			result = append(result, convertCodexAssistantMessages(msg)...)
		case model.MessageKindToolResult:
			if m, ok := convertCodexToolResult(msg); ok {
				result = append(result, m)
			}
		}
	}

	return result
}

func convertCodexUserMessage(msg model.Message) (any, bool) {
	type contentItem struct {
		Type     string `json:"type"`
		Text     string `json:"text,omitempty"`
		ImageURL string `json:"image_url,omitempty"`
	}

	content := make([]contentItem, 0, len(msg.Content))
	for _, p := range msg.Content {
		switch p.Type {
		case model.ContentPartTypeText:
			if p.Text != "" {
				content = append(content, contentItem{Type: "input_text", Text: p.Text})
			}
		case model.ContentPartTypeImage:
			if p.Image != nil && len(p.Image.Data) > 0 {
				content = append(content, contentItem{
					Type:     "input_image",
					ImageURL: fmt.Sprintf("data:%s;base64,%s", p.Image.MimeType, base64.StdEncoding.EncodeToString(p.Image.Data)),
				})
			}
		}
	}

	if len(content) == 0 {
		return nil, false
	}

	return map[string]any{
		"role":    "user",
		"content": content,
	}, true
}

func convertCodexAssistantMessages(msg model.Message) []any {
	result := make([]any, 0, 1+len(msg.ToolCallRequests))

	text := extractTextParts(msg.Content)
	if text != "" {
		result = append(result, map[string]any{
			"type":   "message",
			"role":   "assistant",
			"status": "completed",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        text,
				"annotations": []any{},
			}},
		})
	}

	for _, tc := range msg.ToolCallRequests {
		result = append(result, map[string]any{
			"type":      "function_call",
			"call_id":   tc.ID,
			"name":      tc.ToolID,
			"arguments": string(tc.Arguments),
		})
	}

	return result
}

func convertCodexToolResult(msg model.Message) (any, bool) {
	if strings.TrimSpace(msg.ToolCallID) == "" {
		return nil, false
	}

	output := extractTextParts(msg.Content)
	if output == "" {
		output = "(no output)"
	}

	return map[string]any{
		"type":    "function_call_output",
		"call_id": msg.ToolCallID,
		"output":  output,
	}, true
}

func convertCodexResponse(resp codexResponse) model.Message {
	msg := model.Message{
		Kind: model.MessageKindLLM,
		Metadata: &model.MessageMetadata{
			StopReason: convertCodexStopReason(resp.Status),
			Model:      resp.Model,
			Usage:      convertCodexUsage(resp.Usage),
		},
	}

	for _, out := range resp.Output {
		switch out.Type {
		case "message":
			if out.Role != "assistant" {
				continue
			}
			text := extractCodexText(out.Content)
			if text != "" {
				msg.Content = append(msg.Content, model.ContentPart{Type: model.ContentPartTypeText, Text: text})
			}
		case "function_call":
			callID := strings.TrimSpace(out.CallID)
			if callID == "" {
				callID = strings.TrimSpace(out.ID)
			}
			if callID == "" || strings.TrimSpace(out.Name) == "" {
				continue
			}

			args := strings.TrimSpace(out.Arguments)
			if args == "" {
				args = "{}"
			}

			msg.ToolCallRequests = append(msg.ToolCallRequests, model.ToolCallRequest{
				ID:        callID,
				ToolID:    out.Name,
				Arguments: json.RawMessage(args),
			})
		}
	}

	if len(msg.ToolCallRequests) > 0 {
		msg.Metadata.StopReason = model.StopReasonToolUse
	}

	return msg
}

func extractCodexText(content []codexResponseBlock) string {
	parts := make([]string, 0, len(content))
	for _, c := range content {
		switch c.Type {
		case "output_text":
			if strings.TrimSpace(c.Text) != "" {
				parts = append(parts, c.Text)
			}
		case "refusal":
			if strings.TrimSpace(c.Refusal) != "" {
				parts = append(parts, c.Refusal)
			}
		}
	}

	return strings.Join(parts, "")
}

func extractTextParts(parts []model.ContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}

	return strings.Join(texts, "\n")
}

func convertCodexUsage(u *codexUsage) *model.Usage {
	if u == nil {
		return nil
	}

	raw := model.Usage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
	}
	if u.InputTokensDetails != nil {
		raw.CacheReadTokens = u.InputTokensDetails.CachedTokens
	}

	normalized := usageutil.Normalize(raw, true)

	return &normalized
}

func convertCodexStopReason(status string) model.StopReason {
	switch status {
	case "completed":
		return model.StopReasonComplete
	case "incomplete":
		return model.StopReasonMaxTokens
	case "failed", "cancelled":
		return model.StopReasonError
	default:
		return model.StopReasonComplete
	}
}
