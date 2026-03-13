package simple

import (
	"context"
	"fmt"

	"github.com/slok/gosimov/pkg/agent"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

const (
	// defaultContextWindowTokens is the assumed total context window budget.
	// It is intentionally conservative and model-agnostic.
	defaultContextWindowTokens = 200000
	// defaultReserveTokens keeps room for the model response.
	// Matches Pi-mono's reserveTokens default.
	defaultReserveTokens = 16384
	// defaultKeepRecentTokens matches Pi-mono's default keep window.
	defaultKeepRecentTokens = 20000
	// defaultMaxSummaryTokens is 80% of Pi-mono's reserveTokens (16384 * 0.8).
	defaultMaxSummaryTokens = 13107
)

// Config configures the simple LLM compactor.
//
// The compactor uses a dedicated provider for summarization calls so
// callers can choose a different model than the main conversation model.
type Config struct {
	Provider         llm.Provider
	ReserveTokens    int
	KeepRecentTokens int
	MaxSummaryTokens int
}

func (c *Config) defaults() error {
	if c.Provider == nil {
		return fmt.Errorf("provider is required: %w", pkgerrors.ErrNotValid)
	}

	if c.KeepRecentTokens <= 0 {
		c.KeepRecentTokens = defaultKeepRecentTokens
	}

	if c.ReserveTokens <= 0 {
		c.ReserveTokens = defaultReserveTokens
	}

	if c.MaxSummaryTokens <= 0 {
		c.MaxSummaryTokens = defaultMaxSummaryTokens
	}

	return nil
}

type Compactor struct {
	provider         llm.Provider
	reserveTokens    int
	keepRecentTokens int
	maxSummaryTokens int
}

type compactionSplit struct {
	toSummarize []model.Message
	toKeep      []model.Message
}

// New creates a simple force-capable compactor.
func New(cfg Config) (*Compactor, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid compactor config: %w", err)
	}

	return &Compactor{
		provider:         cfg.Provider,
		reserveTokens:    cfg.ReserveTokens,
		keepRecentTokens: cfg.KeepRecentTokens,
		maxSummaryTokens: cfg.MaxSummaryTokens,
	}, nil
}

var _ agentcontext.Compactor = (*Compactor)(nil)

// Compact applies existing checkpoints and, when forced, creates a new checkpoint.
//
// Behavior:
//  1. Always apply the latest existing checkpoint to produce the effective context.
//  2. If forced, summarize older messages and return a new compaction checkpoint.
//  3. If not forced, summarize only when estimated tokens exceed the threshold;
//     otherwise return filtered context only.
func (c *Compactor) Compact(ctx context.Context, messages []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
	effective := effectiveMessages(messages)
	if !c.shouldCreateCheckpoint(ctx, opts.Force, effective) {
		return &agentcontext.CompactResult{Messages: effective}, nil
	}

	split, ok := c.splitForCompaction(effective)
	if !ok {
		return &agentcontext.CompactResult{Messages: effective}, nil
	}

	firstKeptID := firstMessageID(split.toKeep)
	if firstKeptID == "" {
		return nil, fmt.Errorf("first kept message has no id: %w", pkgerrors.ErrNotValid)
	}

	// If there is a previous checkpoint inside the summarized window,
	// feed its summary to the LLM so it can update incrementally.
	previousSummary := latestSummaryText(split.toSummarize)
	summary, usage, err := c.summarize(ctx, split.toSummarize, previousSummary, opts.CustomInstructions)
	if err != nil {
		return nil, fmt.Errorf("summarizing context: %w", err)
	}

	checkpoint := createCheckpoint(summary, firstKeptID, estimateMessagesTokens(effective))
	filtered := prependCheckpoint(split.toKeep, checkpoint)

	return &agentcontext.CompactResult{
		Message:  &checkpoint,
		Messages: filtered,
		Usage:    usage,
	}, nil
}

func effectiveMessages(messages []model.Message) []model.Message {
	return applyLatestCheckpoint(messages)
}

func (c *Compactor) shouldCreateCheckpoint(ctx context.Context, force bool, messages []model.Message) bool {
	if force {
		return true
	}

	return c.shouldCompact(ctx, messages)
}

func (c *Compactor) splitForCompaction(messages []model.Message) (compactionSplit, bool) {
	// Find where to cut the effective context. Everything before cut is summarized,
	// everything from cut onward is kept verbatim.
	cut := findCutIndex(messages, c.keepRecentTokens)
	if cut <= 0 {
		// Nothing to summarize (context already small or cut would remove all history).
		return compactionSplit{}, false
	}

	return compactionSplit{
		toSummarize: copyMessages(messages[:cut]),
		toKeep:      copyMessages(messages[cut:]),
	}, true
}

func (c *Compactor) shouldCompact(ctx context.Context, messages []model.Message) bool {
	if len(messages) == 0 {
		return false
	}

	maxInputTokens := c.contextWindowTokensFromCtx(ctx) - c.reserveTokens
	return estimateMessagesTokens(messages) > maxInputTokens
}

func (c *Compactor) contextWindowTokensFromCtx(ctx context.Context) int {
	info := agent.LLMModelInfoFromCtx(ctx)
	if info != nil && info.ContextWindow > 0 {
		return info.ContextWindow
	}

	return defaultContextWindowTokens
}

// summarize performs the dedicated summary LLM call and extracts text + usage.
func (c *Compactor) summarize(ctx context.Context, messages []model.Message, previousSummary, customInstructions string) (string, model.Usage, error) {
	conversation := serializeMessages(messages)
	prompt := makeSummarizationPrompt(conversation, previousSummary, customInstructions)

	resp, err := c.provider.Call(ctx, llm.Request{
		SystemPrompt: summarizationSystemPrompt,
		Messages: []model.Message{{
			Kind: model.MessageKindUser,
			Content: []model.ContentPart{{
				Type: model.ContentPartTypeText,
				Text: prompt,
			}},
		}},
		Config: llm.RequestConfig{MaxTokens: c.maxSummaryTokens},
	})
	if err != nil {
		return "", model.Usage{}, err
	}

	summary := firstText(resp.Message)
	if summary == "" {
		return "", model.Usage{}, fmt.Errorf("summary response missing content: %w", pkgerrors.ErrLLMError)
	}

	usage := model.Usage{}
	if resp.Message.Metadata != nil && resp.Message.Metadata.Usage != nil {
		usage = *resp.Message.Metadata.Usage
	}

	return summary, usage, nil
}
