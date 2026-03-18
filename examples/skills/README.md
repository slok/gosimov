# skills example

This example shows a custom `skill` tool running with the OpenCode Go provider.

The `skill` tool:

- Lists local skills in its `Description()` using an OpenCode-style `<available_skills>` catalog.
- Loads full skill content on demand when called with `{ "name": "..." }`.

## Run

```bash
go run ./examples/skills --api-key "$OPENCODE_GO_API_KEY"

# with debug logs
go run ./examples/skills --api-key "$OPENCODE_GO_API_KEY" --debug
```

## Local skills path

By default this example reads skills from:

```text
examples/skills/skills
```

Override it with:

```bash
go run ./examples/skills --api-key "$OPENCODE_GO_API_KEY" --skills-dir /path/to/skills
```

## Debug logging

Use `--debug` to print:

- skill-tool internals (discovery, catalog generation, and skill loads) with a `[skills-tool]` prefix.
- tool execution logs from an example-only wrapper with a `[skills-example]` prefix.
- shell logs include the command, and read logs include the file path (plus offset/limit when set).
