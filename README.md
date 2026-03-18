# gosimov

Build embedded agents in Go, with real guardrails.

`gosimov` is a simple Go SDK for building stateful LLM agents with tools.

It is designed for real automation workflows, especially in infra/SRE environments where Go is a common language for CLIs, operators, controllers, and internal platforms.

Build your own agentic developer/operator experience directly in your Go systems: from coding copilots and chat assistants to production-safe infrastructure automation agents.

## Features

- Simple session API (`Prompt`, `Continue`, `Compact`) with minimal setup.
- Built-in tool-calling loop with JSON schema contracts and structured tool results.
- Multi-provider support (Zen, OpenCode Go, OpenAI, ChatGPT/Codex, Anthropic/Claude).
- Optional persistence backends (`memory`, `jsonl`) plus live subscriptions via `subscriber`.
- Context compaction support for long-running conversations.
- Runtime state + usage tracking for observability and UI integration.
- Provider model metadata (`ModelInfo`) including context window and output limits.
- Extensible architecture: swap and customize tools, compactors, providers, context processors, and session/message storage.
- Go-first API that fits infra automation patterns (e.g., Kubernetes and SRE tooling).

## Install

```bash
go get github.com/slok/gosimov
```

## Quickstart (Zen, real provider)

Set your API key:

```bash
export ZEN_API_KEY="<your-key>"
```

Then run this program:

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/slok/gosimov/pkg/agent"
    "github.com/slok/gosimov/pkg/llm/zen"
    "github.com/slok/gosimov/pkg/model"
)

func main() {
    ctx := context.Background()

    provider, err := zen.New(zen.Config{
        TokenSource: zen.NewAPIKeyTokenSource(os.Getenv("ZEN_API_KEY")),
        Model:       "big-pickle",
    })
    if err != nil {
        panic(err)
    }

    repo := memory.NewRepository()

    session, err := agent.NewSession(ctx, agent.SessionConfig{
        Provider:          provider,
        SystemPrompt:      "You are a concise software assistant.",
        SessionRepository: repo,
        MessageRepository: repo,
        TurnMaxIterations: 8,
    })
    if err != nil {
        panic(err)
    }

    result, err := session.Prompt(ctx, []model.ContentPart{{
        Type: model.ContentPartTypeText,
        Text: "Give me 3 practical tips to debug flaky Go tests.",
    }}, agent.PromptOptions{})
    if err != nil {
        panic(err)
    }

    fmt.Println(result.Message.Content[0].Text)

    usage := session.Usage()
    fmt.Printf("tokens: total=%d input=%d output=%d\n", usage.TotalTokens, usage.InputTokens, usage.OutputTokens)
}
```

## Run Full Examples

```bash
# Real provider + tools
go run ./examples/zen --api-key "$ZEN_API_KEY"

# OpenCode Go provider + tools
go run ./examples/opencode-go --api-key "$OPENCODE_GO_API_KEY"

# Fake provider scripted flow (offline)
go run ./examples/simple

# Multi-turn + forced compaction
ZEN_API_KEY="$ZEN_API_KEY" go run ./examples/compaction

# Browser chat UI
go run ./examples/chat --provider zen --api-key "$ZEN_API_KEY"

# Automated PR review (requires gh auth)
go run ./examples/pr-review --api-key "$OPENCODE_GO_API_KEY" --repo owner/repo --pr 123
```

## Tool Usage Example

This pattern wires real tools into a session. The model can request these tools and the agent loop executes them automatically.

```go
workDir := "/tmp/my-workdir"

lsTool, _ := ls.New(ls.Config{CWD: workDir})
readTool, _ := read.New(read.Config{CWD: workDir})
writeTool, _ := write.New(write.Config{CWD: workDir})
shellTool, _ := shell.New(shell.Config{CWD: workDir})

repo := memory.NewRepository()

session, _ := agent.NewSession(ctx, agent.SessionConfig{
    Provider:          provider,
    Tools:             []tool.Tool{lsTool, readTool, writeTool, shellTool},
    SessionRepository: repo,
    MessageRepository: repo,
})

_, _ = session.Prompt(ctx, []model.ContentPart{{
    Type: model.ContentPartTypeText,
    Text: "Create hello.py with a hello world and run it with python3.",
}}, agent.PromptOptions{})
```

See a complete runnable flow in `examples/simple/main.go` and `examples/zen/main.go`.

## Custom Kubernetes Tool (Guardrailed)

In platform/infrastructure automation, you usually do not want to expose a generic shell tool.
Instead, create narrow tools with explicit allowlists and validation.

```go
type kubectlGetPodsTool struct{}

type kubectlGetPodsInput struct {
    Namespace string `json:"namespace" jsonschema:"required,description=Kubernetes namespace to query"`
}

var kubectlGetPodsSchema = schema.MustFromType[kubectlGetPodsInput]()

func (t kubectlGetPodsTool) ID() string          { return "k8s_get_pods" }
func (t kubectlGetPodsTool) Description() string { return "List pods from an allowed namespace." }
func (t kubectlGetPodsTool) Schema() json.RawMessage { return kubectlGetPodsSchema }

func (t kubectlGetPodsTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
    var in kubectlGetPodsInput
    if err := schema.DecodeStrict(args, &in); err != nil {
        return nil, err
    }

    // Guardrail 1: strict namespace allowlist.
    allowed := map[string]bool{"payments-staging": true, "core-staging": true}
    if !allowed[in.Namespace] {
        return nil, fmt.Errorf("namespace %q is not allowed", in.Namespace)
    }

    // Guardrail 2: fixed command shape (no arbitrary shell).
    cmd := exec.CommandContext(ctx, "kubectl", "get", "pods", "-n", in.Namespace, "-o", "name")
    out, err := cmd.CombinedOutput()
    if err != nil {
        return nil, fmt.Errorf("kubectl failed: %s", string(out))
    }

    return &tool.Result{Content: []model.ContentPart{{
        Type: model.ContentPartTypeText,
        Text: string(out),
    }}}, nil
}
```

This gives you deterministic control over what the agent can do in Kubernetes.

## Why This (vs Agent CLIs and MCP)

Agent CLIs and MCP can be useful, but infrastructure workflows usually need tighter control and fewer moving pieces.

- Guardrails by construction: expose only explicit tools (no unrestricted shell by default).
- Domain-safe validation: enforce namespace/cluster/resource policy before execution.
- Deterministic policy paths: approvals, audit logs, RBAC checks, and change windows are always in your code path.
- Lower operational complexity: no extra MCP server lifecycle, routing, and deployment surface if you do not need it.
- Embedded integration: run directly inside Go infra systems (Kubernetes controllers, Terraform providers, Prometheus exporters, platform APIs).
- Keep agentic behavior without giving up platform safety boundaries.

## Common Patterns

### 1) Add tools to the agent

```go
repo := memory.NewRepository()

session, err := agent.NewSession(ctx, agent.SessionConfig{
    Provider:          provider,
    Tools:             []tool.Tool{lsTool, readTool, writeTool, editTool, shellTool},
    SessionRepository: repo,
    MessageRepository: repo,
})
```

The agent loop executes requested tools automatically and feeds `tool_result` messages back to the model.

### 2) Persist and resume sessions

```go
repo, _ := jsonl.New(jsonl.Config{Dir: "/tmp/gosimov-store"})

session, _ := agent.NewSession(ctx, agent.SessionConfig{
    Provider:          provider,
    SessionRepository: repo,
    MessageRepository: repo,
})

loaded, _ := agent.LoadSession(ctx, agent.LoadSessionConfig{
    SessionID:         session.Session().ID,
    Provider:          provider,
    SessionRepository: repo,
    MessageRepository: repo,
})

_, _ = loaded.Continue(ctx, agent.PromptOptions{})
```

Advanced customization: `LoadSessionConfig.Messages` can override repository-preloaded
history when non-empty (for example, pre-trimmed/pre-sanitized messages). Nil and
empty behave the same and load from `MessageRepository`.

`LoadSession` cannot be used to reset a session's history to empty; create a new
session when you want a reset.

Branching: `SessionConfig.Messages` can bootstrap a new session from prior messages.
That initial branched history is persisted.

### 3) Enable context compaction

```go
summaryProvider, _ := zen.New(zen.Config{TokenSource: zen.NewAPIKeyTokenSource(os.Getenv("ZEN_API_KEY")), Model: "big-pickle"})

compactor, _ := simple.New(simple.Config{
    Provider:         summaryProvider,
    KeepRecentTokens: 1200,
})

repo := memory.NewRepository()

session, _ := agent.NewSession(ctx, agent.SessionConfig{
    Provider:          provider,
    Compactor:         compactor,
    SessionRepository: repo,
    MessageRepository: repo,
})

_, _ = session.Compact(ctx) // manual compaction
```

### 4) Observe session runtime state

```go
state := session.State()
fmt.Println(state.Running, state.Operation, state.Turn, state.MessageCount)
```

### 5) Read loaded model metadata

```go
info := provider.ModelInfo()
fmt.Println(info.ID, info.ContextWindow, info.MaxOutputTokens)
```

Useful for context-window math and UX telemetry.

## Notes

- `TurnMaxIterations` protects from infinite tool-call loops.
- `ToolTimeout` sets a per-tool execution timeout (`0` means no timeout).
- `NewSession` and `LoadSession` require both `SessionRepository` and `MessageRepository`.
- Session configuration is immutable after creation; use `PromptOptions` for per-call overrides.
- `Continue(ctx, opts)` requires existing messages in session history.
- Provider constructors validate model IDs and auth config up front.

## What You Can Build

- Interactive coding chat agents with filesystem and shell tools.
- Session exporters (e.g., static HTML conversation reports).
- Session viewers/inspectors for debugging and observability.
- Long-running infra copilots with compaction and persisted history.
- SRE/operator automation assistants that orchestrate Kubernetes/infra workflows.

## More

- `examples/simple/main.go`
- `examples/zen/main.go`
- `examples/compaction/main.go`
- `examples/chat/main.go`
- `examples/chat/README.md`
- `examples/pr-review/main.go`
- `examples/pr-review/README.md`
