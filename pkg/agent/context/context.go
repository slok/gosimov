// Package context provides context management for the agent loop.
//
// Two interfaces handle different concerns:
//
//   - [Compactor] manages context compaction: it decides when to compact,
//     creates compaction checkpoints, and filters messages based on those
//     checkpoints. It may make LLM calls for summarization.
//
//   - [Processor] is a pure transform on the message list before each LLM call,
//     for concerns like message injection, token trimming, or filtering.
//     It must not mutate the input slice or have side effects.
//
// Both run on every LLM call within a turn. The compactor runs first
// (may compact + filter), then the processor transforms the result.
package context

import (
	"context"

	"github.com/slok/gosimov/pkg/model"
)

// CompactOptions controls compaction behavior.
type CompactOptions struct {
	// Force skips threshold checks and always compacts.
	// When false, the compactor decides based on its internal rules
	// (e.g., token count exceeds threshold).
	Force bool
	// CustomInstructions provides optional instructions to focus the summary.
	// For example: "focus on the auth refactor" or "emphasize the API changes".
	// Empty string means use the default summarization prompt.
	CustomInstructions string
}

// CompactResult is returned by [Compactor.Compact].
type CompactResult struct {
	// Message is the compaction checkpoint message created during compaction.
	// Nil if no compaction was needed (threshold not exceeded and Force was false).
	// The caller is responsible for appending this to the conversation history
	// and persisting it.
	Message *model.Message
	// Messages is the filtered message list for the LLM.
	// If compaction occurred, this excludes content covered by the summary.
	// If no compaction occurred, this may still be filtered based on existing
	// compaction checkpoints in the history.
	Messages []model.Message
	// Usage is the token usage from the summarization LLM call.
	// Zero if no compaction was performed.
	Usage model.Usage
}

// Compactor manages context compaction.
//
// On each call, the compactor may:
//  1. Decide the context needs compaction (threshold check, or forced via [CompactOptions.Force]).
//  2. Create a [model.MessageKindCompaction] message with a summary (via a dedicated LLM call).
//  3. Return filtered messages that exclude content covered by the summary.
//
// Implementations must treat input messages as immutable and return a new slice
// when they need to transform ordering or values.
//
// If no compaction is needed, it returns filtered messages based on any
// existing compaction checkpoints in the history.
//
// The compactor is called on every LLM call within a turn (including
// iterations after tool results are appended). It can also be called
// explicitly between turns via [Session.Compact].
type Compactor interface {
	Compact(ctx context.Context, messages []model.Message, opts CompactOptions) (*CompactResult, error)
}

// Processor transforms messages before they are sent to the LLM.
//
// Implementations must not mutate the input slice. Return a new slice
// with the desired messages for the LLM call.
//
// The processor is called on every LLM call within a turn (including
// iterations after tool results are appended), after the compactor.
type Processor interface {
	ProcessContext(ctx context.Context, messages []model.Message) ([]model.Message, error)
}

// NoopCompactor is a [Compactor] that passes messages through unchanged.
// Used as the default when no compactor is configured.
type NoopCompactor struct{}

// Compact implements [Compactor]. It never compacts and returns all messages unchanged.
func (NoopCompactor) Compact(_ context.Context, messages []model.Message, _ CompactOptions) (*CompactResult, error) {
	return &CompactResult{Messages: messages}, nil
}
