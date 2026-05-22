package mcptool

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewToolSet(t *testing.T) {
	tests := map[string]struct {
		config func(t *testing.T) ToolSetConfig
		check  func(t *testing.T, got *ToolSet, err error)
	}{
		"Missing session should fail.": {
			config: func(_ *testing.T) ToolSetConfig {
				return ToolSetConfig{Namespace: "demo"}
			},
			check: func(t *testing.T, _ *ToolSet, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "session is required")
			},
		},

		"Discovered tools should be wrapped with namespace.": {
			config: func(t *testing.T) ToolSetConfig {
				session := newTestSession(t, func(server *mcp.Server) {
					server.AddTool(&mcp.Tool{
						Name:        "greet",
						Description: "Greets a person.",
						InputSchema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`),
					}, func(_ context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
						var args struct {
							Name string `json:"name"`
						}
						require.NoError(t, json.Unmarshal(req.Params.Arguments, &args))
						return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "Hello " + args.Name}}}, nil
					})
				})

				return ToolSetConfig{Session: session, Namespace: "demo"}
			},
			check: func(t *testing.T, got *ToolSet, err error) {
				require.NoError(t, err)
				require.Len(t, got.Tools(), 1)

				tool := got.Tools()[0]
				assert.Equal(t, "demo_greet", tool.ID())
				assert.Equal(t, "Greets a person.", tool.Description())
				assert.JSONEq(t, `{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`,
					string(tool.Schema()))

				result, err := tool.Execute(context.Background(), json.RawMessage(`{"name":"gosimov"}`))
				require.NoError(t, err)
				require.Len(t, result.Content, 1)
				assert.Equal(t, "Hello gosimov", result.Content[0].Text)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := NewToolSet(context.Background(), test.config(t))
			test.check(t, got, err)
		})
	}
}

func TestWrappedToolID(t *testing.T) {
	tests := map[string]struct {
		namespace string
		name      string
		expID     string
	}{
		"Simple names should join with underscore.": {
			namespace: "demo",
			name:      "greet",
			expID:     "demo_greet",
		},
		"Invalid characters should be normalized.": {
			namespace: "mcp.linear",
			name:      "search/documentation",
			expID:     "mcp_linear_search_documentation",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, test.expID, wrappedToolID(test.namespace, test.name))
		})
	}
}

func TestToolExecute(t *testing.T) {
	tests := map[string]struct {
		session func(t *testing.T) MCPClient
		check   func(t *testing.T, gotText string, gotErr error)
	}{
		"MCP tool error should become gosimov tool error.": {
			session: func(t *testing.T) MCPClient {
				return newTestSession(t, func(server *mcp.Server) {
					server.AddTool(&mcp.Tool{
						Name:        "broken",
						InputSchema: json.RawMessage(`{"type":"object"}`),
					}, func(_ context.Context, _ *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
						return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "something failed"}}}, nil
					})
				})
			},
			check: func(t *testing.T, _ string, gotErr error) {
				require.Error(t, gotErr)
				assert.Contains(t, gotErr.Error(), "something failed")
			},
		},

		"Image content should be preserved.": {
			session: func(t *testing.T) MCPClient {
				return newTestSession(t, func(server *mcp.Server) {
					server.AddTool(&mcp.Tool{
						Name:        "image",
						InputSchema: json.RawMessage(`{"type":"object"}`),
					}, func(_ context.Context, _ *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
						return &mcp.CallToolResult{Content: []mcp.Content{&mcp.ImageContent{Data: []byte("png"), MIMEType: "image/png"}}}, nil
					})
				})
			},
			check: func(t *testing.T, gotText string, gotErr error) {
				require.NoError(t, gotErr)
				assert.Equal(t, "image/png", gotText)
			},
		},

		"Structured content should fall back to JSON text.": {
			session: func(t *testing.T) MCPClient {
				return newTestSession(t, func(server *mcp.Server) {
					server.AddTool(&mcp.Tool{
						Name:        "structured",
						InputSchema: json.RawMessage(`{"type":"object"}`),
					}, func(_ context.Context, _ *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
						return &mcp.CallToolResult{StructuredContent: map[string]any{"message": "hello"}}, nil
					})
				})
			},
			check: func(t *testing.T, gotText string, gotErr error) {
				require.NoError(t, gotErr)
				assert.JSONEq(t, `{"message":"hello"}`, gotText)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			toolSet, err := NewToolSet(context.Background(), ToolSetConfig{Session: test.session(t), Namespace: "demo"})
			require.NoError(t, err)

			result, gotErr := toolSet.Tools()[0].Execute(context.Background(), json.RawMessage(`{}`))
			gotText := ""
			if result != nil && len(result.Content) > 0 {
				if result.Content[0].Image != nil {
					gotText = result.Content[0].Image.MimeType
				} else {
					gotText = result.Content[0].Text
				}
			}

			test.check(t, gotText, gotErr)
		})
	}
}

func newTestSession(t *testing.T, register func(server *mcp.Server)) MCPClient {
	t.Helper()

	ctx := context.Background()
	server := mcp.NewServer(&mcp.Implementation{Name: "test-server", Version: "v1.0.0"}, nil)
	register(server)

	serverTransport, clientTransport := mcp.NewInMemoryTransports()
	serverSession, err := server.Connect(ctx, serverTransport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = serverSession.Close() })

	client := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "v1.0.0"}, nil)
	clientSession, err := client.Connect(ctx, clientTransport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = clientSession.Close() })

	return clientSession
}
