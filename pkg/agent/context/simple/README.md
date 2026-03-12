# simple compactor

This package provides a minimal, deterministic, LLM-backed context compactor for `gosimov` sessions.

It is designed to be easy to reason about and easy to evolve.

The current defaults and heuristics are based on Pi's compaction approach and operating profile (`https://pi.dev`).

## Goals

- Keep long conversations within model input limits.
- Preserve useful recent context while replacing older history with a checkpoint summary.
- Stay provider-agnostic and simple (no tokenizer dependencies).
- Support both automatic and manual (forced) compaction.

## Core concepts

- **Compaction checkpoint**: a `model.MessageKindCompaction` message containing:
  - summary text in `Message.Content`
  - boundary metadata in `Message.Compaction.FirstKeptID`
  - estimated pre-compaction tokens in `Message.Compaction.TokensBefore`
- **Effective context**: message list after applying the latest valid checkpoint.
- **Keep window**: estimated-token budget for recent messages that are kept verbatim.

## Behavior

`Compactor.Compact(ctx, messages, opts)` has 2 modes:

- **Non-forced** (`opts.Force == false`)
  - always applies latest checkpoint filtering first
  - creates a new checkpoint only when estimated input tokens exceed threshold
- **Forced** (`opts.Force == true`)
  - always attempts checkpoint creation (unless there is nothing compactable)

## High-level flow

1. Build **effective messages** by applying the latest valid checkpoint.
2. Decide if a new checkpoint should be created.
3. Split effective messages into:
   - `toSummarize` (older portion)
   - `toKeep` (recent portion)
4. Summarize `toSummarize` with a dedicated provider.
5. Create a compaction checkpoint message.
6. Return filtered list: `[checkpoint] + toKeep`.

## Checkpoint application model

When filtering by checkpoint:

- pick the newest valid compaction message (must have `FirstKeptID`)
- find `FirstKeptID` by ID in current message list
- if missing/invalid, return original messages unchanged
- output shape is:
  - latest checkpoint message
  - all messages from `FirstKeptID` onward
  - no duplicate if checkpoint is inside kept range

This makes compaction robust to persisted/reloaded histories where indices may shift.

## Token estimation and cut strategy

This package uses a simple estimate:

- message tokens ~= `text_chars/4`
- includes tool-call `ToolID` and JSON arguments
- minimum of 1 token when chars > 0

Automatic compaction threshold:

- `estimated_tokens(messages) > contextWindowTokens - reserveTokens`

Cut index selection:

- walk backward accumulating estimated tokens until `keepRecentTokens` is reached
- the cut is the first kept message index
- if cut lands on `tool_result`, move backward to avoid splitting tool-call/result context

## Summarization prompt design

Prompt construction has 3 pieces:

- `<conversation> ... </conversation>` transcript of messages to summarize
- optional `<previous-summary> ... </previous-summary>` for incremental updates
- optional `Custom instructions: ...`

Template mode:

- **initial template** when there is no previous summary
- **update template** when previous summary exists

The system prompt explicitly tells the model to output only structured summary content.

## File layout

- `simple.go`: public config + compactor orchestration
- `checkpoint.go`: checkpoint discovery/application/creation helpers
- `tokens.go`: token estimation and cut logic
- `serialize.go`: transcript serialization helpers
- `prompt.go`: summary system prompt + text template assembly
- `doc.go`: package-level behavior summary

## Why this design

- **Small surface area**: easy to audit and test.
- **Deterministic heuristics**: predictable behavior across providers.
- **Incremental summaries**: can carry forward prior checkpoint summaries.
- **Clear composition**: checkpoint logic, token math, and serialization are separated.

## Limitations and trade-offs

- Token estimate is approximate; different providers tokenize differently.
- Summary quality depends on model and prompt adherence.
- Compaction is message-level; no semantic chunking or retrieval.

## Testing strategy

The package uses two layers:

- **Public (black-box) tests** in `simple_test` package for API behavior and regressions.
- **Internal (white-box) tests** in `simple` package for helper math, checkpoint logic, serialization, and prompt templating.

Together these protect refactors while keeping behavior explicit.
