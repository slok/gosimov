# Chat Example

This example starts an HTTP server with a browser chat UI that uses gosimov sessions, tools, and message subscriptions.

## Features

- Prompt an LLM from the browser.
- Receive persisted messages in real time via SSE.
- Let the agent execute tools (`ls`, `read`, `write`, `edit`, `shell`) autonomously.
- Reload the page and load previously persisted sessions.
- Use simple context compaction (plus manual compact button in the UI).
- Export the selected session as a static HTML file.
- Stop a running prompt/compaction operation from the UI.
- Inspect live SDK-backed session status (`running`, `operation`, `turn`, usage).

## HTTP API

- `GET /api/sessions/{id}/status`: returns the current in-memory SDK session state.

## Run

```bash
go run ./examples/chat --provider zen --api-key <key>

go run ./examples/chat --provider opencode-go --api-key <key>

go run ./examples/chat --provider openai --api-key <key>

go run ./examples/chat --provider anthropic --api-key <key>

# OAuth file flow (Codex)
go run ./examples/chat --provider codex --auth-file /tmp/gosimov-chat-auth.json

# OAuth file flow (Claude Pro/Max)
go run ./examples/chat --provider claude --auth-file /tmp/gosimov-chat-auth.json
```

Then open `http://localhost:8080`.

Optional flags:

- `--addr` (default `:8080`)
- `--provider` (default `zen`; one of `zen`, `opencode-go`, `openai`, `codex`, `anthropic`, `claude`)
- `--api-key` (required unless `--auth-file` is set)
- `--auth-file` (OAuth credentials file path; only for `codex`/`claude` providers)
- `--model` (default depends on `--provider`)
- `--system-prompt` (optional)
- `--max-iterations` (default `100`)
- `--max-history-messages` (default `60`, used when loading existing sessions)
- `--compaction-keep-recent-tokens` (default `1200`)
- `--compaction-summary-model` (default same as `--model`)
- `--store-dir` (default `/tmp/gosimov-chat-store`)
- `--work-dir` (default `/tmp/gosimov-chat-work`)
