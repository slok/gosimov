// Package context provides context management for the agent loop.
//
// Two interfaces handle different concerns:
//
//   - [Compactor] manages context compaction: it decides when to compact,
//     creates compaction checkpoints, and may make LLM calls for summarization.
//
//   - [Processor] is a pure transform on the message list before each LLM call,
//     for concerns like message injection, token trimming, or filtering.
//     It must not mutate the input slice or have side effects.
//
// Both run on every LLM call within a turn. The compactor runs first,
// then the processor transforms the runtime context messages.
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
	// SummaryMessage is the compaction checkpoint message created during compaction.
	// It is nil if no new checkpoint was created (e.g., compaction not needed or forced off).
	SummaryMessage *model.Message
	// Usage is the token usage from the summarization LLM call.
	// Zero if no compaction was performed.
	Usage model.Usage
}

// Compactor manages context compaction.
//
// On each call, the compactor may:
//  1. Decide the context needs compaction (threshold check, or forced via [CompactOptions.Force]).
//  2. Create a [model.MessageKindCompaction] message with a summary (via a dedicated LLM call).
//
// Implementations must treat input messages as immutable.
//
// The compactor is called on every LLM call within a turn (including
// iterations after tool results are appended). It can also be called
// explicitly between turns via [Session.Compact].
type Compactor interface {
	Compact(ctx context.Context, messages []model.Message, opts CompactOptions) (*CompactResult, error)
}

// NoopCompactor is a [Compactor] that never creates checkpoints.
// Used as the default when no compactor is configured.
type NoopCompactor struct{}

// Compact implements [Compactor]. It never creates checkpoint messages.
func (NoopCompactor) Compact(_ context.Context, _ []model.Message, _ CompactOptions) (*CompactResult, error) {
	return &CompactResult{}, nil
}
