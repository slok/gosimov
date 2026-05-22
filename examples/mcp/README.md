# MCP Example

This example shows how to use remote MCP tools with gosimov without adding MCP support to the core library.

Flow:

1. Connect to a remote MCP server over Streamable HTTP.
2. Discover available MCP tools with `tools/list`.
3. Wrap each discovered MCP tool as a normal gosimov `tool.Tool`.
4. Run a normal gosimov session with a real OpenCode Go provider.
5. When the LLM calls a wrapped tool, gosimov translates that into MCP `tools/call`.

## Why This Is An Example

This keeps MCP integration outside `pkg/` until the shape is proven useful.

The core library still sees only normal gosimov tools.

## Architecture

- `main.go`: real session wiring, provider setup, remote MCP HTTP connection, optional headers.
- `mcptool/`: MCP discovery and tool wrapper implementation.

The important boundary is:

- gosimov agent loop <-> gosimov `tool.Tool`
- wrapper <-> MCP client session

The LLM does not speak MCP directly.

## Limitations

- Discovery happens once at startup.
- Only MCP tools are supported here, not MCP prompts or resources.
- Tool IDs are normalized to provider-safe names (`[a-zA-Z0-9_-]`).
- The wrapper maps MCP text and image content. Unsupported MCP content types are omitted with a text notice.
- The example uses HTTP request/response mode (`DisableStandaloneSSE: true`), so it does not rely on server-initiated notifications.

## Usage

```bash
go run ./examples/mcp \
  --api-key "$(tr -d '\n' < ./bin/opencode-api.key)" \
  --mcp-url "https://mcp.linear.app/mcp" \
  --mcp-header "Authorization: Bearer $(tr -d '\n' < ./bin/linear-api.key)" \
  --prompt "List my Linear teams and summarize them briefly."
```

Optional flags:

- `--model`
- `--base-url`
- `--mcp-namespace`
- repeated `--mcp-header "Name: Value"`
