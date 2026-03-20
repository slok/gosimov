# Gosimov Project Guide

**Purpose:** Project-specific patterns and guidance for AI agents working on Gosimov.

**Scope:** This file extends `.agents/base.md` with Gosimov-specific structure and workflows.

**Maintenance:** Keep in sync with codebase. Update when architecture or key patterns change.

---

## Project Overview

Gosimov is a Go SDK for building agentic LLM workflows. It provides the building blocks to add LLM-powered agents to CLI tools, services, and infrastructure tooling.

**Inspiration:** Pi-mono (TypeScript) for its simplicity and clean architecture. Also studied Shelley (Go) and OpenCode (TypeScript).

**Philosophy:** Simple, composable, idiomatic Go. No magic frameworks. Build incrementally, document and test alongside code.

## Package Layout

```
github.com/slok/gosimov/
  pkg/
    model/        # Core data types (Message, Turn, Usage, etc.)
    store/        # Storage interfaces (SessionRepository, MessageRepository)
    store/memory/ # In-memory storage implementation (implements both)
    store/jsonl/  # JSONL file-based storage implementation (implements both)
    store/subscriber/ # MessageRepository wrapper that emits message-stored events
    agent/         # Agent loop (runTurn) and Session
    agent/context/ # Context interfaces (Compactor, Processor)
    agent/context/simple/ # Simple LLM compactor (force-only checkpoint creation)
    llm/           # LLM provider interface and types
    llm/fake/      # Fake LLM providers for testing
    llm/openai/    # OpenAI-compatible Chat Completions provider
    pkgerrors/     # Shared sentinel errors
    tool/          # Tool interface
    tool/toolmock/ # Generated mock for Tool interface
    tool/ls/       # ls tool — lists directory contents
    tool/read/     # read tool — reads file contents
    tool/write/    # write tool — writes file contents
    tool/edit/     # edit tool — search-and-replace edits on files
    tool/shell/    # shell tool — executes shell commands + Executor interface + CMDExecutor
    tool/shell/shellmock/ # Generated mock for Executor interface
  internal/
    utils/id/       # ULID generation
    utils/file/     # File utilities (path sanitization, truncation, detection, read-write FS)
  examples/
    chat/         # Browser chat server (HTTP + SSE) using subscriber-backed message persistence
    compaction/   # Multi-turn example with Zen provider + simple compactor + forced Session.Compact
    pprof/        # Deterministic fake-provider workload runner for pprof and benchmark profiling
    openai-oauth/ # OAuth auth-code+PKCE example with file-backed credentials and auto refresh
    pr-review/    # CI-ready PR reviewer using dedicated GitHub tools (no generic shell)
    simple/       # Minimal usage example (fake provider, all 5 tools)
    zen/          # End-to-end example with OpenCode Zen API
    viewer/       # HTTP server for browsing JSONL sessions as HTML conversations
```

## Core Model Types

All core data types live in `pkg/model/`.

### Message

`Message` is the fundamental unit. Messages are stored flat, not grouped into turns.

Fields populated based on `Kind`:

| Kind | Fields used |
|------|-------------|
| `MessageKindUser` | `Content` |
| `MessageKindLLM` | `Content`, `ToolCallRequests`, `Metadata` |
| `MessageKindToolResult` | `Content`, `ToolCallID`, `IsError` |
| `MessageKindCompaction` | `Content` (summary text), `Compaction` |

### Naming Conventions

These naming decisions are intentional — follow them consistently:

| We use | Not | Why |
|--------|-----|-----|
| `MessageKindLLM` / `"llm"` | `"assistant"` | It's the LLM responding, not an abstract assistant. Provider adapters translate to API-specific terms. |
| `MessageKind` | `Role` | It's a kind of message, not a role. `tool_result` doesn't have a "role." |
| `ToolCallRequest` | `ToolCall` | It's a request from the LLM to call a tool, not the execution itself. |
| `ToolID` | `Name` | If it identifies something, it's an ID. Names are for display. `"bash"` is an ID, `"Bash"` is a name. |
| `ContentPartType` | `ContentType` | Unambiguous — it's the type of a `ContentPart`, not a generic content type. |
| `StopReasonComplete` | `StopReasonEndTurn` | Neutral. Not tied to any provider's terminology. |
| `ImageData` | `ImageContent` | It holds data (bytes + mime type). "Content" is overloaded in the message context. |

### ContentPart

`ContentPart` is purely displayable content: text and images. Tool calls and tool result metadata are separate fields on `Message`, not content part variants.

This keeps `ContentPart` simple and separates concerns:
- `Content []ContentPart` — what you display
- `ToolCallRequests []ToolCallRequest` — what you execute (LLM messages only)
- `ToolCallID` / `IsError` — linking back to a request (tool result messages only)

### Turn

`Turn` is a **derived/computed type**, not stored. It groups a user message with its following LLM and tool result messages. Use `TurnsFromMessages()` to compute turns from a flat message list.

Storage deals only with individual messages. Turns are for the agent loop and UI.

### ContextUsage

`ContextUsage` is a **derived/computed type**, not stored. It represents context window utilization based on the most recent LLM call's reported token usage. Use `ContextUsageFromMessages()` to extract it from a flat message list.

`TotalInputTokens` (`InputTokens + CacheReadTokens`) is the actual context size the provider processed. Compare it with `LLMModelInfo.ContextWindow` to compute utilization percentage. Returns zero if the last LLM message has no usage metadata — does not fall back to earlier messages.

### Session

`Session` is a domain entity that identifies a multi-turn conversation. It is purely identity — messages, usage, and other state are associated to a session via its `ID` but are not stored inside this struct.

| Field | Type | Description |
|-------|------|-------------|
| `ID` | `string` | ULID, assigned at creation time by `NewSession()`. |
| `CreatedAt` | `time.Time` | Timestamp when the session was created. |

The runtime `Session` in `pkg/agent/` holds a `model.Session` internally and passes it through to `runTurn` via `turnConfig`. The store layer uses this identity to associate persisted data with sessions.

### Usage

`Usage` tracks token consumption for a single LLM call. Attached to LLM messages via `MessageMetadata`.

### StopReason

Indicates why the LLM stopped generating. Each value is a distinct end condition callers can branch on (similar in spirit to sentinel errors):

| Value | Meaning |
|-------|---------|
| `""` (none) | Zero value — no stop reason set (in-progress or unfinished) |
| `complete` | Model finished naturally |
| `max_tokens` | Hit output token limit, response truncated |
| `tool_use` | Model wants to call tools, agent loop should continue |
| `error` | LLM call failed |
| `aborted` | Request cancelled (context, user abort, timeout) |

## LLM Provider

The LLM provider interface lives in `pkg/llm/`.

### Interface

`Call` is blocking — sends messages to the LLM and waits for the full response. Streaming will be added later as a separate method. The agent loop logic doesn't change — streaming only adds real-time visibility into the response as it arrives.

`Request` contains `SystemPrompt`, `Messages`, and `Config`. Tool definitions are passed to providers at construction time via their `Config` (not per-request).

`Response` contains a `Message` with `Kind`, `Content`, `ToolCallRequests`, and `Metadata` set. The caller sets `ID` and `CreatedAt`.

`RequestConfig` currently has `MaxTokens` (0 means provider default). Additional fields (temperature, etc.) are added when needed.

### Fake Providers

`pkg/llm/fake/` provides test implementations:

- `fake.NewProvider(fn)` — configurable provider that delegates to a function.
- `fake.NewEchoProvider()` — convenience constructor that echoes the last user message back as an LLM response. Built on top of `NewProvider`.

Both return `*fake.Provider` which implements `llm.Provider`.

### OpenAI-Compatible Provider

`pkg/llm/openai/` implements `llm.Provider` for any OpenAI-compatible Chat Completions API (POST `/chat/completions`). Works with OpenAI, OpenCode Zen, Azure OpenAI, Ollama, vLLM, etc.

**Config:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `APIKey` | Conditional | — | Static bearer token for authentication. Set exactly one of `APIKey` or `TokenSource`. |
| `TokenSource` | Conditional | — | Dynamic bearer token source (`Token(ctx)`) for OAuth/rotating auth. Set exactly one of `APIKey` or `TokenSource`. |
| `BaseURL` | No | `https://api.openai.com/v1` | API base URL. For Zen: `https://opencode.ai/zen/v1` |
| `Model` | Yes | — | Model ID (e.g. `"glm-5-free"`, `"gpt-4o"`) |
| `Tools` | No | `nil` | `[]tool.Tool` — converted to OpenAI tool definitions at construction |
| `ProviderID` | No | `"openai"` | Provider name in response metadata |
| `Client` | No | `http.DefaultClient` | HTTP client for API calls |

**OAuth helpers:**

- `OAuthTokenSource` implements `TokenSource` and supports OAuth Authorization Code + PKCE and refresh-token flows.
- `AuthorizationRequest(state)` returns authorization URL + state + PKCE verifier.
- `ExchangeAuthorizationCode(ctx, code, codeVerifier)` exchanges code and persists credentials.
- `Token(ctx)` returns valid access token, auto-refreshing and persisting when expired.
- `FileCredentialsStore` persists credentials in a JSON file (`0600` permissions).

**Design decisions:**
- **Tools at construction time** — `[]tool.Tool` is passed in `Config`, converted to OpenAI format once in `New()`. The tool set is fixed for the provider's lifetime.
- **No external dependencies** — uses only `net/http` + `encoding/json`. No OpenAI Go SDK.
- **Private API types** — all OpenAI request/response types (`chatRequest`, `chatResponse`, etc.) are unexported in `convert.go`. Only `Config`, `Provider`, and `New` are public.
- **Stop reason mapping** — `"stop"` → `Complete`, `"length"` → `MaxTokens`, `"tool_calls"` → `ToolUse`, `"content_filter"` → `Error`, unknown → `Complete`.
- **Usage mapping** — `InputTokens` is canonical non-cached input across providers. For OpenAI-compatible APIs, `prompt_tokens_details.cached_tokens` maps to `CacheReadTokens` and is subtracted from `prompt_tokens` to derive `InputTokens`.
- **Error handling** — non-200 status codes return `ErrLLMError` with the API error message parsed from JSON when possible.
- **Single text content** — user messages with one text part use string content (`"content": "hello"`), multi-part messages use array content (`"content": [{...}]`). This maximizes compatibility across providers.
- **Image encoding** — images are encoded as `data:<mime>;base64,<data>` URLs in `image_url` content parts.

**Message mapping (gosimov → OpenAI):**

| Gosimov Kind | OpenAI Role | Content handling |
|-------------|-------------|-----------------|
| `MessageKindUser` | `"user"` | Text → string or array, images → `image_url` |
| `MessageKindLLM` | `"assistant"` | Text → content, tool calls → `tool_calls` |
| `MessageKindToolResult` | `"tool"` | Text content joined, `tool_call_id` set |
| System prompt | `"system"` | Prepended as first message |

**Testing:**
- `convert_test.go` — unit tests for all conversion functions (pure mapping, no HTTP).
- `openai_test.go` — `httptest.NewServer`-based tests for the full provider (request/response round-trip, error handling, header verification).

### ChatGPT Codex Responses Provider

`pkg/llm/openai/` also provides a ChatGPT Codex backend provider via `NewCodexResponses` (POST `/codex/responses`).

**Config highlights:**

- Auth: exactly one of `APIKey` or `TokenSource`.
- Default base URL: `https://chatgpt.com/backend-api`.
- Requires JWT token with `https://api.openai.com/auth.chatgpt_account_id` claim (sent as `chatgpt-account-id` header).

**Design notes:**
- Blocking `Call` uses non-stream responses (`stream=false`) and maps output message/function_call items into gosimov messages/tool calls.
- Uses `OpenAI-Beta: responses=experimental` and `originator` headers.

## Tool

The tool interface lives in `pkg/tool/`.

`Tool` is an interface with `ID()`, `Description()`, `Schema()`, and `Execute()`. Tools are what the LLM can request to run during the agent loop.

`Result` contains `Content` (displayable output) and represents a successful execution. If a tool fails, it returns a Go `error` — the agent loop sends `err.Error()` to the LLM as an error tool result (`IsError=true` on the message). This is idiomatic Go: `error` means failure, `Result` means success.

Mocks are generated with mockery into `pkg/tool/toolmock/`. Run `make go-gen` to regenerate after interface changes.

### Tool Implementation Pattern

Concrete tools live under `pkg/tool/<toolid>/`. Each tool follows this pattern:

- **`Config` struct** with a `defaults() error` method for validation and defaults.
- **`New(config Config) (*Tool, error)`** constructor.
- **`fs.FS`** for filesystem access — pluggable via `Config.FS`, defaults to `os.DirFS(cwd)`. Tests use `fstest.MapFS`.
- **Path sanitization** — all user-provided paths are cleaned, must be relative, and cannot escape the working directory via `..`.
- **Output truncation** — uses `internal/utils/file` for byte/line limits. Notices are appended when limits are hit.
- **Error handling** — invalid input or execution failures return Go errors. The agent loop sends `err.Error()` to the LLM as an error tool result. The LLM sees the error and decides what to do.

### ls Tool

`pkg/tool/ls/` — lists directory contents.

**Config:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `CWD` | Yes | — | Working directory for path resolution |
| `FS` | No | `os.DirFS(CWD)` | Filesystem to read from (`fs.FS`) |
| `EntryLimit` | No | `500` | Max entries to return |
| `MaxBytes` | No | `50KB` | Max byte size of output |

**Schema parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `"."` | Directory to list, relative to CWD |
| `limit` | integer | 500 | Max entries (capped by `EntryLimit`) |

**Behavior:**
- Returns entries sorted alphabetically (case-insensitive), one per line.
- Directories get a `/` suffix.
- Includes hidden files (dotfiles).
- Empty directory returns `"(empty directory)"`.
- Appends `[N entries limit reached]` and/or `[50.0KB limit reached]` notices when limits are hit.
- Rejects absolute paths, `..` traversal, and non-directory targets with descriptive errors.

### read Tool

`pkg/tool/read/` — reads file contents.

**Config:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `CWD` | Yes | — | Working directory for path resolution |
| `FS` | No | `os.DirFS(CWD)` | Filesystem to read from (`fs.FS`) |
| `MaxLines` | No | `2000` | Max lines to return |
| `MaxBytes` | No | `50KB` | Max byte size of output |

**Schema parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | — (required) | File to read, relative to CWD |
| `offset` | integer | `1` | Line number to start from (1-indexed) |
| `limit` | integer | — | Max lines to read (before truncation limits) |

**Behavior:**
- Reads entire file into memory, then applies offset/limit in-memory.
- **Content detection** via magic bytes (no external dependencies):
  - **Images** (PNG, JPEG, GIF, WebP): returned as `ImageData` content part + text note. Offset/limit are ignored.
  - **Binary** (null bytes in first 4KB): rejected with error `"cannot read binary file: <path>"`.
  - **Text**: existing line-based reading with offset/limit/truncation.
- Offset is 1-indexed (matches how editors/humans think about line numbers).
- User `limit` is applied first, then `MaxLines`/`MaxBytes` truncation.
- Appends actionable continuation notices:
  - Truncation hit: `[Showing lines X-Y of Z. Use offset=N to continue.]`
  - User limit applied with more content: `[N more lines in file. Use offset=N to continue.]`
  - Byte truncation adds `(50.0KB limit)` to the notice.
- **mtime**: all successful results include file modification time as a text notice (`[Modified: <RFC3339>]`). For images, included in the text note. The edit tool accepts this mtime as an optional parameter for staleness detection.
- Errors: missing path, file not found, offset beyond EOF, absolute paths, `..` traversal, binary files.

### write Tool

`pkg/tool/write/` — writes file contents.

**Config:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `CWD` | Yes | — | Working directory for path resolution |
| `FS` | No | `file.NewOSReadWriteFS(CWD)` | Writable filesystem (`file.ReadWriteFS`) |

**Schema parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | File to write, relative to CWD |
| `content` | string | Yes | Content to write |

**Behavior:**
- Creates parent directories automatically (`MkdirAll`).
- Overwrites existing files unconditionally.
- Reports `"Created <path> (N bytes)"` or `"Overwrote <path> (N bytes)"`.
- Rejects absolute paths and `..` traversal via `SanitizePath`.
- No mtime/staleness check — that's an edit tool concern.

Uses `file.ReadWriteFS` (shared with the edit tool) for filesystem operations.

### edit Tool

`pkg/tool/edit/` — performs search-and-replace edits on files.

**Config:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `CWD` | Yes | — | Working directory for path resolution |
| `FS` | No | `file.NewOSReadWriteFS(CWD)` | Writable filesystem (`file.ReadWriteFS`) |

**Schema parameters:**

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | Yes | — | File to edit, relative to CWD |
| `old_text` | string | Yes | — | Exact text to find in the file |
| `new_text` | string | Yes | — | Replacement text |
| `replace_all` | boolean | No | `false` | Replace all occurrences instead of requiring uniqueness |
| `mtime` | string | No | — | Expected file mtime (RFC3339) for staleness detection |

**Behavior:**
- Finds `old_text` exactly using `strings.Index` (no fuzzy matching).
- By default, requires a unique match — errors if `old_text` appears more than once.
- `replace_all: true` replaces all occurrences without uniqueness check.
- Empty `new_text` deletes the matched text.
- **CRLF normalization:** file content and search text are normalized to LF for matching. If the original file used CRLF, line endings are restored after replacement.
- **mtime staleness detection:** if `mtime` is provided (RFC3339), the file's current mtime is compared (truncated to seconds). Mismatch rejects the edit. The LLM gets mtime from the read tool's `[Modified: <RFC3339>]` notice and passes it here.
- Rejects: empty `old_text` (suggests write tool), identical old/new text, absolute paths, `..` traversal.
- **Result includes unified diff:** successful edits return a unified diff of the changes (3 context lines, standard `---`/`+++`/`@@` format). Computed on normalized (LF) content before CRLF restoration. No external diff library — uses `file.FormatUnifiedDiff` which leverages the known replacement positions.
- Error messages guide the LLM to retry (e.g., "provide more surrounding context" for multiple matches).

Uses `file.ReadWriteFS` (shared with the write tool) for filesystem operations.

### shell Tool

`pkg/tool/shell/` — executes shell commands. Contains the tool, the `Executor` interface, and the default `CMDExecutor` implementation.

**Tool Config:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `CWD` | Yes | — | Working directory for commands |
| `Executor` | No | `CMDExecutor` | Shell executor (`Executor` interface) |
| `DefaultTimeout` | No | `120s` | Per-command timeout when not specified by LLM |
| `MaxLines` | No | `2000` | Max output lines (tail truncation) |
| `MaxBytes` | No | `50KB` | Max output bytes (tail truncation) |

**Schema parameters:**

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `command` | string | Yes | — | Shell command to execute |
| `timeout` | integer | No | `120` | Timeout in seconds |

**Behavior:**
- Each command spawns a new process — **stateless**, no state persists across calls.
- Output is combined stdout+stderr with **tail truncation** (keeps the end, where errors typically appear).
- Appends notices for: timeout (`[Command timed out]`), non-zero exit code (`[Exit code: N]`), truncation (`[Showing last N of M lines]`).
- Empty output returns `"(no output)"`.
- Executor is pluggable via `Executor` interface — mockable for tests, wrappable for Docker/SSH.

**Executor interface (`executor.go`):**
```go
type Executor interface {
    Exec(ctx context.Context, command string, cwd string, timeout time.Duration) (*Result, error)
}
```

**CMDExecutor** — default implementation using `exec.Command`:
- Stateless: each `Exec` call spawns `exec.Command(shell, "-c", command)` with `cmd.Dir = cwd`.
- stdout/stderr captured via `bytes.Buffer` (no temp files needed).
- Timeout via `context.WithTimeout` + process group kill (`SIGKILL` to `-pid`).
- `GIT_EDITOR=true` injected in env.
- Shell binary resolved: `/bin/bash` → `/usr/bin/bash` → PATH bash → PATH sh → `/bin/sh`.
- Non-zero exit codes extracted from `exec.ExitError`, returned as `Result.ExitCode` (not as Go errors).

**CMDExecutorConfig:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `ShellPath` | No | auto-resolved | Shell binary path |
| `Env` | No | `os.Environ()` | Environment for spawned processes |

**Mock:** `pkg/tool/shell/shellmock/` — generated via mockery. Used by shell tool tests to avoid real process spawning.

## File Utilities

`internal/utils/file/` groups file-related helpers used by tools. One package, multiple files by concern.

### Path Sanitization (`path.go`)

- `SanitizePath(path) (string, error)` — cleans, validates, and normalizes a user-provided path.
- Rejects absolute paths and `..` traversal (returns descriptive errors).
- Returns forward-slash paths for `fs.FS` compatibility.
- Used by `ls`, `read`, `write`, and `edit` tools.

### Text Truncation (`truncate.go`)

- `TruncateHead(content, TruncateOpts) (string, TruncateResult)` — keeps the first portion that fits within byte/line limits. Truncates at line boundaries when possible. Used by read and ls tools.
- `TruncateTail(content, TruncateOpts) (string, TruncateResult)` — keeps the last portion that fits within byte/line limits. Truncates at line boundaries when possible. Used by the bash tool (errors and results appear at the end of output).
- `FormatSize(bytes) string` — human-readable byte formatting (`"50.0KB"`, `"1.2MB"`).
- `TruncateOpts` — `MaxBytes int`, `MaxLines int` (0 = no limit for either).
- `TruncateResult` — `Truncated bool`, `OriginalBytes`, `OriginalLines`, `KeptBytes`, `KeptLines`.

### Content Detection (`detect.go`)

- `DetectContent(data) DetectResult` — inspects raw file data using magic bytes and binary heuristics.
- `DetectResult` — `Kind` (`ContentKindText`, `ContentKindImage`, `ContentKindBinary`) and `MimeType` (set for images).
- Image detection: magic byte signatures for PNG, JPEG, GIF, WebP. No external dependencies.
- Binary detection: null byte check in the first 4KB.
- Used by the `read` tool to branch between text reading, image returning, and binary rejection.

### Unified Diff Formatter (`diff.go`)

- `FormatUnifiedDiff(path, oldContent, newContent, replacements, contextLines) string` — produces a unified diff from known replacement positions. No diff algorithm needed — leverages known byte offsets.
- `Replacement` — describes one replacement by `Offset`, `OldLen`, `NewLen` in the original content.
- `DefaultContextLines` = 3 (standard unified diff default).
- Handles: single replacements, multiple replacements (`replace_all`), deletions, multi-line expansions/contractions, hunk merging when context overlaps.
- Helper exports: `SplitLines(s)`, `ByteOffsetToLine(content, offset)`.
- Used by the `edit` tool to include a diff in the result content.

### Read-Write Filesystem (`rwfs.go`)

- `ReadWriteFS` interface — abstracts filesystem operations for components that read and write files. Go's `fs.FS` is read-only, so components that modify the filesystem need this.
- Methods: `Stat(path)`, `ReadFile(path)`, `ReadDir(path)`, `MkdirAll(path)`, `WriteFile(path, data)`, `AppendFile(path, data)`.
- `NewOSReadWriteFS(root)` — default implementation wrapping `os` functions rooted at a directory.
- Used by the `write` tool, `edit` tool, and `jsonl` store.
- Consumers define their own narrower interfaces if they don't need all methods.
- Tests provide in-memory implementations (`memFS`) with panicking stubs for unused methods.

## Store

The store layer lives in `pkg/store/`.

### Interfaces

Two separate interfaces in `pkg/store/store.go`:

- `SessionRepository` — `CreateSession`, `GetSession`, `ListSessions`.
- `MessageRepository` — `StoreMessages`, `ListMessages`.

Both list operations use generic cursor-based pagination:

```go
type ListOpts struct {
    Cursor string // Opaque. Empty = start from beginning.
    Limit  int    // 0 = no limit (return all).
}

type ListResult[T any] struct {
    Items      []T
    NextCursor string // Empty = no more items.
}
```

The cursor is an opaque string — each implementation encodes whatever it needs. Callers must not parse or construct cursors.

### Memory Implementation

`pkg/store/memory/` provides `Repository`, which implements both `SessionRepository` and `MessageRepository`. Data is stored in-memory using maps protected by `sync.RWMutex`. All data is lost when the process exits.

- `NewRepository() *Repository`
- `ListSessions` returns sessions newest-first (reverse insertion order).
- `ListMessages` returns messages in insertion order.
- Cursor is an index encoded as a decimal string.

### JSONL File-Based Implementation

`pkg/store/jsonl/` provides `Repository`, which implements both `SessionRepository` and `MessageRepository` using JSONL files on disk. Each session is a single self-contained file: `<dir>/<session_id>.jsonl`.

- `New(cfg Config) (*Repository, error)` — creates the store directory if needed.
- Append-only file format: first line is the session header, remaining lines are messages.
- Deleting a file removes all its data. No separate index or metadata files.
- Safe for concurrent use within the same process (`sync.RWMutex`).

**Config:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `Dir` | Yes | — | Directory for session files. Created automatically if missing. |
| `FS` | No | `file.NewOSReadWriteFS(Dir)` | Filesystem implementation (`file.ReadWriteFS`) |

Uses the shared `file.ReadWriteFS` interface from `internal/utils/file/` — the same one used by write and edit tools. The default `osReadWriteFS` implementation is rooted at `Dir`, so all paths inside the store are relative. Tests use `t.TempDir()` with the default FS.

**File format:**

Each `.jsonl` file is:
```
{"type":"session","id":"<ulid>","created_at":"<rfc3339>"}
{"type":"message","id":"<ulid>","kind":"user","content":[...],"created_at":"<rfc3339>"}
{"type":"message","id":"<ulid>","kind":"llm","content":[...],"metadata":{...},"created_at":"<rfc3339>"}
...
```

Line types are identified by the `"type"` field: `"session"` or `"message"`.

**JSONL line types (all unexported in `convert.go`):**

| Type | Fields | Description |
|------|--------|-------------|
| `sessionLine` | `type`, `id`, `created_at` | Session header (always first line) |
| `messageLine` | `type`, `id`, `kind`, `content`, `tool_call_requests`, `tool_call_id`, `is_error`, `metadata`, `created_at` | Message entry |
| `contentPartLine` | `type`, `text`, `image` | Content part (text or image) |
| `imageDataLine` | `data`, `mime_type` | Image binary data (base64-encoded by `json.Marshal`) |
| `toolCallLine` | `id`, `tool_id`, `arguments` | Tool call request |
| `metadataLine` | `usage`, `stop_reason`, `model`, `provider` | LLM response metadata |
| `usageLine` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `total_tokens`, `reasoning_tokens` | Token usage |
| `compactionDataLine` | `first_kept_id`, `tokens_before` | Compaction checkpoint data |

**Design decisions:**
- **Images stored inline as base64** — same as Pi-mono. No external storage, no stripping. `json.Marshal` handles `[]byte` → base64 automatically.
- **Per-message token usage stored** — for debugging, context window management, and analytics.
- **Corrupt lines are skipped** — `parseMessages` silently skips unparseable or wrong-type lines for resilience.
- **Filename is always `<session_id>.jsonl`** — no customization. Users wanting something different implement their own repository.
- **Cursor is index-based** — same as memory store (decimal string encoding of slice index).
- **`ListSessions` sorts newest-first** — reads all `.jsonl` files, parses first line of each, sorts by `CreatedAt` descending. Skips files with mismatched filename/ID or corrupt headers.
- **`StoreMessages` batches into single append** — all messages are serialized into one buffer and written in a single `AppendFile` call for atomicity.
- **Generic `paginate` helper** — shared by all list methods. Same cursor/limit semantics as memory store.

**Testing:**
- `convert_test.go` — round-trip conversion tests (session, all message kinds, JSON marshaling).
- `jsonl_test.go` — full store interface tests using `t.TempDir()` as real filesystem (create/get/list sessions, store/list messages, pagination, file contents verification, image round-trip).

### Sentinel Errors

- `ErrNotFound` — session or messages not found.
- `ErrAlreadyExists` — duplicate session creation.

### Subscriber Wrapper

`pkg/store/subscriber/` provides a lightweight `store.MessageRepository` decorator that emits events after successful `StoreMessages` calls.

- `New(Config)` wraps an existing `store.MessageRepository`.
- `Subscribe(ctx, SubscribeOpts)` returns `<-chan MessageStoredEvent` and auto-unsubscribes/close on context cancellation.
- Optional replay (`SubscribeOpts.Replay=true`) replays historical messages for one session via paginated `ListMessages`, then switches to live events.
- Delivery is non-blocking best-effort per subscriber; on full buffers, oldest events are dropped to keep newest updates flowing.
- This is process-local notification, not a distributed pub/sub mechanism.

### Session Integration

`SessionConfig` requires both `SessionRepository` and `MessageRepository`:

- `NewSession(ctx, cfg)` calls `CreateSession` to persist the session identity.
- Messages are persisted **individually as they are produced** (per-message persistence, like Pi-mono):
  1. `Prompt` persists the user message eagerly (before calling `runTurn`).
  2. Each LLM response is persisted immediately after being stamped with ID/timestamp.
  3. Each tool result is persisted individually after execution.
  4. `Continue` persists only the turn's messages (no user message to persist).
- The `onMessages` callback on `turnConfig` is the mechanism — `Session.persistMessages` is wired as the callback.
- Repository errors are fatal — they propagate as errors from the caller.

## Agent Loop

The agent loop lives in `pkg/agent/`.

### runTurn (private)

`runTurn(ctx, turnConfig) (*TurnResult, error)` is the internal engine that executes one turn. It's private — users interact with the `Session` type, which calls `runTurn` internally.

`turnConfig` carries the `model.Session` identity alongside provider, messages, tools, and max iterations. This means everything inside the loop has access to session context for storage, logging, or events.

`TurnResult` is the only public type from the turn layer (it's the return type of `Session.Prompt()` and `Session.Continue()`).

### runCompaction (private)

`runCompaction(ctx, compactionConfig) (*CompactResult, error)` is the internal engine for between-turn compaction. It's the compaction counterpart to `runTurn` — both live in `agent.go`, both handle persistence via the `onMessages` callback, and both are called by `Session`.

```
Session.Prompt    → runTurn
Session.Continue  → runTurn
Session.Compact   → runCompaction
```

`compactionConfig` carries the compactor, messages, onMessages callback, and `CompactOptions`. The function calls `compactor.Compact()`, persists the compaction message if created, and returns the result. The caller (`Session.Compact`) updates in-memory state (append message, aggregate usage).

This keeps compaction logic in one place (the turn runner layer) instead of duplicating it between `runTurn` (in-turn compaction) and `Session.Compact` (between-turn compaction).

### Loop Logic

1. Call the LLM with conversation history.
2. Stamp the response with ID and CreatedAt.
3. Check StopReason:
   - `Complete`, `MaxTokens`, `None` (no metadata) → return result.
   - `Error` → return error.
   - `Aborted` → return error.
   - `ToolUse` → execute tool calls, append results, loop back to step 1.
4. If TurnMaxIterations > 0 and exceeded → return error.

### Tool Execution

- Tools are indexed by ID for O(1) lookup at session creation time (`buildToolIndex`). Duplicate tool IDs are rejected at `NewSession`/`LoadSession`, not at turn time.
- Tool not found → `MessageKindToolResult` with `IsError=true` and descriptive text.
- `Execute()` returns error → `MessageKindToolResult` with `IsError=true` and `err.Error()` as message.
- `Execute()` returns `(*Result, nil)` → success result with `IsError=false`.
- All tool errors are fed back to the LLM (never abort the loop on tool errors).

### TurnResult

`TurnResult` contains:
- `Message` — the final LLM response that ended the turn.
- `Messages` — all new messages generated during the turn (LLM responses + tool results).
- `Usage` — aggregated token usage across all LLM calls in the turn.

## Session

The session layer lives in `pkg/agent/` alongside `runTurn`.

### Session Struct

`Session` is the stateful wrapper around the turn runner layer (`runTurn` and `runCompaction`). It manages the conversation history, accumulates usage, and enforces single-concurrency.

Each `Session` holds a `model.Session` (domain entity with ID and CreatedAt) that gives it stable identity. This identity flows into `runTurn` via `turnConfig.session`, making it available to any subsystem inside the loop.

### SessionConfig

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `Provider` | Yes | — | LLM provider to call |
| `SystemPrompt` | No | `""` | System instruction for the LLM |
| `Tools` | No | `nil` | Available tools for the LLM to call |
| `ToolTimeout` | No | `0` | Per-tool execution timeout. `0` means no timeout. |
| `TurnMaxIterations` | No | `0` (no limit) | Per-turn LLM call limit. 0 means unlimited. |
| `SessionRepository` | Yes | — | Session identity repository. New sessions are persisted on creation. |
| `MessageRepository` | Yes | — | Message repository used for per-message persistence during turns. |
| `Messages` | No | `nil` | Advanced: preload initial in-memory history (e.g., branching). Usage is reconstructed from these messages, and initial messages are persisted on creation. |
| `Compactor` | No | `NoopCompactor` | Manages compaction inside the turn loop and via `Session.Compact()`. |
| `ContextProcessor` | No | `nil` | If set, transforms messages before each LLM call (after compactor). |

### LoadSessionConfig

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `SessionID` | Yes | — | Existing persisted session ID to load. |
| `Provider` | Yes | — | LLM provider to call |
| `SystemPrompt` | No | `""` | System instruction for the LLM |
| `Tools` | No | `nil` | Available tools for the LLM to call |
| `ToolTimeout` | No | `0` | Per-tool execution timeout. `0` means no timeout. |
| `TurnMaxIterations` | No | `0` (no limit) | Per-turn LLM call limit. 0 means unlimited. |
| `SessionRepository` | Yes | — | Repository used to load session identity by ID. |
| `MessageRepository` | Yes | — | Repository used to preload persisted message history when `Messages` is empty. |
| `Messages` | No | `nil` | Advanced: when non-empty, overrides repository preload and is used as in-memory history instead. Empty behaves like nil (load from repository). |
| `Compactor` | No | `NoopCompactor` | Manages compaction inside the turn loop and via `Session.Compact()`. |
| `ContextProcessor` | No | `nil` | If set, transforms messages before each LLM call (after compactor). |

### API

| Method | Description |
|--------|-------------|
| `NewSession(ctx, cfg) (*Session, error)` | Creates a new session with ULID and timestamp. Persists session identity and can preload/persist initial history via `SessionConfig.Messages`. |
| `LoadSession(ctx, cfg) (*Session, error)` | Loads an existing persisted session identity. Uses `LoadSessionConfig.Messages` when non-empty; otherwise preloads from `MessageRepository`. |
| `Prompt(ctx, []ContentPart, opts PromptOptions) (*TurnResult, error)` | Builds a user message, appends it, runs a turn. `PromptOptions` can override `SystemPrompt` and `TurnMaxIterations` for that call. |
| `Continue(ctx, opts PromptOptions) (*TurnResult, error)` | Runs a turn from current messages (retries). `PromptOptions` can override `SystemPrompt` and `TurnMaxIterations` for that call. |
| `Compact(ctx) (*CompactResult, error)` | Delegates to `runCompaction` with `Force: true`. Appends the compaction message + aggregates usage if created. Returns `ErrSessionBusy` if a turn is running. |
| `Session() model.Session` | Returns the session identity (ID and creation time). |
| `State() SessionState` | Returns a thread-safe runtime snapshot (`running`, `operation`, `turn`, `message_count`, identity, usage). |
| `Messages() []Message` | Returns a copy of the conversation history. |
| `Usage() Usage` | Returns aggregated usage across all turns. |

`PromptOptions` fields:

- `SystemPrompt` — optional per-call system prompt override. Empty means use the session's `SystemPrompt`.
- `TurnMaxIterations` — optional per-call max-iterations override. `0` means use the session's `TurnMaxIterations`.

### Concurrency

Only one `Prompt`, `Continue`, or `Compact` can be active at a time. Concurrent calls return `ErrSessionBusy`. All methods are mutex-protected.

`Session.State()` exposes the active operation kind while running: `prompt`, `continue`, `compact`.

### Error Behavior

If `runTurn` fails after `Prompt` builds and appends the user message, the user message remains in history (it was already appended). The caller can retry with `Continue()`.

## Context Management

Context management lives in `pkg/agent/context/`. Two interfaces handle different concerns within the agent turn loop.

### Compactor

`Compactor` has a single method: `Compact(ctx, messages, CompactOptions) → (*CompactResult, error)`. It owns the full compaction lifecycle: deciding when to compact, creating compaction checkpoint messages, and filtering messages based on existing checkpoints.

**The caller (agent loop or `Session.Compact`) is responsible for appending the compaction message to history, persisting it, and aggregating usage.** The compactor only creates the message — it doesn't mutate session state.

The compactor runs **first** on every LLM call within a turn. When no `Compactor` is configured, `NoopCompactor` is used (passthrough, set in both `SessionConfig.defaults()` and `turnConfig.defaults()`). This avoids branching in the loop and nil checks in `Session.Compact()`.

**CompactOptions:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Force` | `bool` | `false` | Skip threshold checks, always compact. The turn loop passes `false` (threshold-based), `Session.Compact()` passes `true`. |
| `CustomInstructions` | `string` | `""` | Focus the summary (e.g., "focus on auth refactor"). Empty = default summarization prompt. |

**CompactResult:**

| Field | Type | Description |
|-------|------|-------------|
| `Message` | `*model.Message` | Compaction checkpoint message (nil if no compaction needed). |
| `Messages` | `[]model.Message` | Filtered messages for the LLM. Excludes content covered by any compaction checkpoints. |
| `Usage` | `model.Usage` | Token usage from the summarization LLM call (zero if no compaction). |

`pkg/agent/context/simple` provides the first real compactor implementation. Design:
1. `Force=false`: creates a checkpoint when token threshold is exceeded, otherwise only filters by existing checkpoints.
2. `Force=true`: always attempts checkpoint creation via dedicated LLM summarization call.
3. Keeps recent context using Pi-mono style `keepRecentTokens` (chars/4 estimation).
4. Never cuts at `tool_result` boundaries.
5. Uses structured markdown summarization prompts (Goal/Constraints/Progress/Decisions/Next Steps/Critical Context).
6. Uses plain-text conversation serialization (`[User]: ...`, `[LLM]: ...`, `[Tool Result]: ...`) for summarization inputs.
7. Uses `contextWindowTokens - reserveTokens` trigger budget for automatic compaction.

### Processor

`Processor` has a single method: `ProcessContext(ctx, messages) (messages, error)`. It's a pure transform on the message list — no side effects, no history mutation. Used for concerns like message injection, token trimming, or filtering.

The processor runs **after** the compactor on every LLM call within a turn.

### Turn Loop Flow

```
for each iteration:
    1. compactor.Compact(allMessages, CompactOptions{}) → compactResult
    2. If compactResult.Message != nil → append to allMessages, persist, aggregate usage
    3. llmMessages = compactResult.Messages
    4. processor.ProcessContext(llmMessages) → llmMessages      // pure transform
    5. LLM call with llmMessages
    6. If tool use → execute tools, append to allMessages, go to 1
```

### Compaction Model

`MessageKindCompaction` ("compaction") is a first-class message kind. It represents a compaction checkpoint — a summary of older messages.

| Field | Type | Description |
|-------|------|-------------|
| `Content` | `[]ContentPart` | Summary text of compacted messages |
| `Compaction.FirstKeptID` | `string` | ID of first message kept (not summarized) |
| `Compaction.TokensBefore` | `int` | Token count before compaction (analytics only) |

Compaction messages are **stored persistently** in the JSONL file (append-only, nothing deleted). Old messages remain in storage but are skipped when building LLM context.

Current implementation note: `context/simple` supports both automatic threshold-triggered compaction and forced compaction.

### What's NOT Here Yet

- Improved token estimation from provider usage metadata (today uses chars/4 heuristic).
- Steering messages (mid-turn interrupts).
- Follow-up messages (post-turn continuation).
- Agent lifecycle/streaming events (message deltas, turn hooks, etc.).
- Streaming support.
- Additional `RequestConfig` fields (temperature, top_p, etc.) as providers need them.

## Testing

- Table-driven tests using `map[string]struct{...}` (see `base.md`)
- Run tests: `make test`
- Run lint: `make check` (requires golangci-lint, runs in CI container)
- Run all: `make ci`

### Integration Tests

Integration tests live in `tests/integration/session/` and test full sessions against the OpenCode Go provider. They are gated by environment variables and skipped when not configured.

**Environment variables:**

| Var | Required | Default | Description |
|-----|----------|---------|-------------|
| `GOSIMOV_INTEGRATION` | Yes | — | Must be `"true"` to run integration tests |
| `INTEGRATION_OPENCODE_GO_API_KEY` | Yes | — | OpenCode Go API key |
| `INTEGRATION_OPENCODE_GO_MODEL` | No | `minimax-m2.5` | Model for conversation |
| `OPENCODEGO_INTEGRATION_SUMMARY_MODEL` | No | same as model | Model for compaction summarization |
| `OPENCODEGO_INTEGRATION_TIMEOUT` | No | `3m` | Per-prompt timeout (Go duration) |

**Run integration tests:**

```sh
GOSIMOV_INTEGRATION=true INTEGRATION_OPENCODE_GO_API_KEY=<key> go test -v -count=1 ./tests/integration/session/
```

**Test coverage:**

- `TestSimpleResponse` — basic LLM response, metadata, token usage
- `TestToolUsageWriteEditRead` — multi-step tool orchestration (write, edit, read)
- `TestToolUsageListDirectory` — ls tool with pre-existing files
- `TestCompaction` / `TestCompactionResultFields` / `TestCompactionNoopWhenNotForced` — compaction flow, fields, forced vs noop
- `TestSessionJSONLExport` — JSONL file structure verification
- `TestSessionLoadFromJSONL` — session load from persisted JSONL
- `TestTokenUsageAccumulation` — multi-turn token tracking

**Design decisions:**

- No shell tool — only file tools (ls, read, write, edit) to avoid arbitrary code execution risk in CI
- No build tags — env var gating is simpler (`t.Skipf` when `GOSIMOV_INTEGRATION != "true"`)
- No `t.Parallel()` — free tier rate limits cause flakes with concurrent requests
- Assertions are structural (message kinds, metadata presence, token counts > 0), not on exact LLM text
- Config follows the `NewConfig(t)` helper pattern (see `tests/integration/session/helpers_test.go`)

## CI

- **Check job:** golangci-lint in `golangci/golangci-lint:v2.10.1-alpine` container
- **Unit test job:** `actions/setup-go` with Go >= 1.25, runs `make test`
- **Integration test job:** `actions/setup-go` with Go >= 1.25, runs `go test -v -count=1 ./tests/integration/session/` when `OPENCODE_GO_API_KEY` secret is configured (uses `INTEGRATION_OPENCODE_GO_API_KEY` env var)
- **PR review job:** `.github/workflows/pr-review.yml` triggers on `@gosimov-review` mentions and runs `go run ./examples/pr-review` with trusted associations only (`OWNER`, `MEMBER`, `COLLABORATOR`)
- Triggered on push and pull request

## Code Generation

- Mocks: `make go-gen` runs mockery. Config in `.mockery.yml`.
- Mock packages follow `{package}mock/mocks.go` naming (e.g., `pkg/tool/toolmock/mocks.go`).
- Generated files are committed to the repo (not gitignored).
- Never edit generated files directly.

## Dependencies

- `github.com/oklog/ulid/v2` — ULID generation for message/turn IDs
- `github.com/stretchr/testify` — test assertions and mocking (test dependency)
