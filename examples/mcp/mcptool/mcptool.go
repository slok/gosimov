// Package mcptool adapts MCP tools into gosimov tools.
package mcptool

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
)

var defaultSchema = json.RawMessage(`{"type":"object"}`)

var invalidToolIDChars = regexp.MustCompile(`[^a-zA-Z0-9_-]+`)

// MCPClient is the subset of an MCP client session required by the adapter.
type MCPClient interface {
	ListTools(ctx context.Context, params *mcp.ListToolsParams) (*mcp.ListToolsResult, error)
	CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error)
}

// ToolSetConfig configures a discovered MCP tool set.
type ToolSetConfig struct {
	// Session is the active MCP client session used for discovery and execution.
	Session MCPClient
	// Namespace prefixes each discovered tool ID. Example: "demo" -> "demo.greet".
	Namespace string
}

func (c *ToolSetConfig) defaults() error {
	if c.Session == nil {
		return fmt.Errorf("session is required")
	}

	c.Namespace = strings.TrimSpace(c.Namespace)
	if c.Namespace == "" {
		return fmt.Errorf("namespace is required")
	}

	return nil
}

// ToolSet is a discovered collection of MCP-backed gosimov tools.
type ToolSet struct {
	tools []tool.Tool
}

// NewToolSet discovers MCP tools and returns them wrapped as gosimov tools.
func NewToolSet(ctx context.Context, cfg ToolSetConfig) (*ToolSet, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid mcp tool set config: %w", err)
	}

	var (
		tools  []tool.Tool
		cursor string
		seen   = map[string]struct{}{}
	)

	for {
		result, err := cfg.Session.ListTools(ctx, &mcp.ListToolsParams{Cursor: cursor})
		if err != nil {
			return nil, fmt.Errorf("listing mcp tools: %w", err)
		}

		for _, mt := range result.Tools {
			wrapped, err := newTool(cfg.Session, cfg.Namespace, mt)
			if err != nil {
				return nil, fmt.Errorf("wrapping mcp tool: %w", err)
			}

			if _, ok := seen[wrapped.ID()]; ok {
				return nil, fmt.Errorf("duplicate wrapped tool id %q", wrapped.ID())
			}
			seen[wrapped.ID()] = struct{}{}
			tools = append(tools, wrapped)
		}

		if result.NextCursor == "" {
			break
		}

		cursor = result.NextCursor
	}

	return &ToolSet{tools: tools}, nil
}

// Tools returns a copy of the wrapped tool slice.
func (s *ToolSet) Tools() []tool.Tool {
	result := make([]tool.Tool, len(s.tools))
	copy(result, s.tools)

	return result
}

type mcpTool struct {
	session     MCPClient
	id          string
	remoteName  string
	description string
	schema      json.RawMessage
}

func newTool(session MCPClient, namespace string, mt *mcp.Tool) (*mcpTool, error) {
	if mt == nil {
		return nil, fmt.Errorf("tool is nil")
	}

	name := strings.TrimSpace(mt.Name)
	if name == "" {
		return nil, fmt.Errorf("tool name is required")
	}

	description := strings.TrimSpace(mt.Description)
	if description == "" {
		description = fmt.Sprintf("Calls MCP tool %q.", name)
	}

	schema, err := normalizeSchema(mt.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("normalize input schema for %q: %w", name, err)
	}

	return &mcpTool{
		session:     session,
		id:          wrappedToolID(namespace, name),
		remoteName:  name,
		description: description,
		schema:      schema,
	}, nil
}

func (t *mcpTool) ID() string { return t.id }

func (t *mcpTool) Description() string { return t.description }

func (t *mcpTool) Schema() json.RawMessage { return t.schema }

func (t *mcpTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	callArgs, err := decodeArguments(args)
	if err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	result, err := t.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      t.remoteName,
		Arguments: callArgs,
	})
	if err != nil {
		return nil, fmt.Errorf("calling mcp tool %q: %w", t.remoteName, err)
	}

	content, errText := convertResultContent(result)
	if result.IsError {
		if errText == "" {
			errText = fmt.Sprintf("mcp tool %q failed", t.remoteName)
		}
		return nil, fmt.Errorf("%s", errText)
	}

	return &tool.Result{Content: content}, nil
}

func normalizeSchema(schema any) (json.RawMessage, error) {
	if schema == nil {
		return defaultSchema, nil
	}

	b, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}

	if len(b) == 0 || string(b) == "null" {
		return defaultSchema, nil
	}

	return json.RawMessage(b), nil
}

func decodeArguments(args json.RawMessage) (map[string]any, error) {
	if len(args) == 0 {
		return map[string]any{}, nil
	}

	var result map[string]any
	if err := json.Unmarshal(args, &result); err != nil {
		return nil, err
	}

	if result == nil {
		result = map[string]any{}
	}

	return result, nil
}

func convertResultContent(result *mcp.CallToolResult) ([]model.ContentPart, string) {
	if result == nil {
		return []model.ContentPart{model.NewContentText("(no output)")}, ""
	}

	content := make([]model.ContentPart, 0, len(result.Content)+1)
	notices := make([]string, 0)

	for _, part := range result.Content {
		switch p := part.(type) {
		case *mcp.TextContent:
			content = append(content, model.NewContentText(p.Text))
		case *mcp.ImageContent:
			content = append(content, model.NewContentImage(p.Data, p.MIMEType))
		default:
			notices = append(notices, fmt.Sprintf("unsupported MCP content type %T omitted", part))
		}
	}

	if len(content) == 0 && result.StructuredContent != nil {
		if b, err := json.Marshal(result.StructuredContent); err == nil {
			content = append(content, model.NewContentText(string(b)))
		}
	}

	if len(notices) > 0 {
		content = append(content, model.NewContentText("["+strings.Join(notices, "; ")+"]"))
	}

	if len(content) == 0 {
		content = append(content, model.NewContentText("(no output)"))
	}

	return content, flattenText(content)
}

func flattenText(parts []model.ContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type == model.ContentPartTypeText {
			text := strings.TrimSpace(part.Text)
			if text != "" {
				texts = append(texts, text)
			}
		}
	}

	return strings.Join(texts, "\n")
}

func wrappedToolID(namespace string, remoteName string) string {
	parts := []string{sanitizeToolIDPart(namespace), sanitizeToolIDPart(remoteName)}
	filtered := make([]string, 0, len(parts))
	for _, part := range parts {
		if part != "" {
			filtered = append(filtered, part)
		}
	}

	return strings.Join(filtered, "_")
}

func sanitizeToolIDPart(part string) string {
	part = strings.TrimSpace(part)
	if part == "" {
		return ""
	}

	part = invalidToolIDChars.ReplaceAllString(part, "_")
	part = strings.Trim(part, "_")

	for strings.Contains(part, "__") {
		part = strings.ReplaceAll(part, "__", "_")
	}

	return part
}
