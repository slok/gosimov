// Package simple provides a minimal LLM-backed context compactor.
//
// The compactor supports two modes:
//  1. Non-forced: apply existing compaction checkpoints and auto-compact
//     when estimated tokens exceed the configured threshold.
//  2. Forced: create a new checkpoint by summarizing older messages.
//
// Checkpoint creation uses a dedicated summarization provider and returns a
// MessageKindCompaction checkpoint plus the filtered message list.
//
// This implementation intentionally keeps the algorithm simple:
//   - chars/4 token estimation
//   - keep recent token window
//   - avoid cutting at tool-result boundaries
//
// It is designed as the first practical compactor and can be evolved later
// with model-aware tokenization and automatic threshold-triggered checkpointing.
package simple
