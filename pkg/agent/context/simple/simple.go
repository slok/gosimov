package simple

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/slok/gosimov/internal/utils/id"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/conventions"
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
	Provider            llm.Provider
	ContextWindowTokens int
	ReserveTokens       int
	KeepRecentTokens    int
	MaxSummaryTokens    int
}

func (c *Config) defaults() error {
	if c.Provider == nil {
		return fmt.Errorf("provider is required: %w", pkgerrors.ErrNotValid)
	}

	if c.KeepRecentTokens <= 0 {
		c.KeepRecentTokens = defaultKeepRecentTokens
	}

	if c.ContextWindowTokens <= 0 {
		c.ContextWindowTokens = defaultContextWindowTokens
	}

	if c.ReserveTokens <= 0 {
		c.ReserveTokens = defaultReserveTokens
	}

	if c.ContextWindowTokens <= c.ReserveTokens {
		return fmt.Errorf("context window tokens must be greater than reserve tokens: %w", pkgerrors.ErrNotValid)
	}

	if c.MaxSummaryTokens <= 0 {
		c.MaxSummaryTokens = defaultMaxSummaryTokens
	}

	return nil
}

type Compactor struct {
	provider            llm.Provider
	contextWindowTokens int
	reserveTokens       int
	keepRecentTokens    int
	maxSummaryTokens    int
}

// New creates a simple force-capable compactor.
func New(cfg Config) (*Compactor, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid compactor config: %w", err)
	}

	return &Compactor{
		provider:            cfg.Provider,
		contextWindowTokens: cfg.ContextWindowTokens,
		reserveTokens:       cfg.ReserveTokens,
		keepRecentTokens:    cfg.KeepRecentTokens,
		maxSummaryTokens:    cfg.MaxSummaryTokens,
	}, nil
}

var _ agentcontext.Compactor = (*Compactor)(nil)

// Compact applies existing checkpoints and, when forced, creates a new checkpoint.
//
// Behavior:
//  1. Always apply the latest existing checkpoint to produce the effective context.
//  2. If not forced, return filtered context only (no new summary call).
//  3. If forced, summarize older messages and return a new compaction checkpoint.
func (c *Compactor) Compact(ctx context.Context, messages []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
	effective := applyLatestCheckpoint(messages)
	if !opts.Force && !c.shouldCompact(effective) {
		return &agentcontext.CompactResult{Messages: effective}, nil
	}

	// Find where to cut the effective context. Everything before cut is summarized,
	// everything from cut onward is kept verbatim.
	cut := findCutIndex(effective, c.keepRecentTokens)
	if cut <= 0 {
		// Nothing to summarize (context already small or cut would remove all history).
		return &agentcontext.CompactResult{Messages: effective}, nil
	}

	toSummarize := copyMessages(effective[:cut])
	toKeep := copyMessages(effective[cut:])

	firstKeptID := firstMessageID(toKeep)
	if firstKeptID == "" {
		return nil, fmt.Errorf("first kept message has no id: %w", pkgerrors.ErrNotValid)
	}

	// If there is a previous checkpoint inside the summarized window,
	// feed its summary to the LLM so it can update incrementally.
	summary, usage, err := c.summarize(ctx, toSummarize, latestSummaryText(toSummarize), opts.CustomInstructions)
	if err != nil {
		return nil, fmt.Errorf("summarizing context: %w", err)
	}

	checkpoint := model.Message{
		ID:        id.NewULID(conventions.IDPrefixCompaction),
		Kind:      model.MessageKindCompaction,
		CreatedAt: time.Now(),
		Content: []model.ContentPart{{
			Type: model.ContentPartTypeText,
			Text: summary,
		}},
		Compaction: &model.CompactionData{
			FirstKeptID:  firstKeptID,
			TokensBefore: estimateMessagesTokens(effective),
		},
	}

	filtered := make([]model.Message, 0, 1+len(toKeep))
	filtered = append(filtered, checkpoint)
	filtered = append(filtered, toKeep...)

	return &agentcontext.CompactResult{
		Message:  &checkpoint,
		Messages: filtered,
		Usage:    usage,
	}, nil
}

func (c *Compactor) shouldCompact(messages []model.Message) bool {
	if len(messages) == 0 {
		return false
	}

	maxInputTokens := c.contextWindowTokens - c.reserveTokens
	return estimateMessagesTokens(messages) > maxInputTokens
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

// applyLatestCheckpoint applies the newest valid checkpoint in the history.
//
// Result shape is:
//   - latest checkpoint message
//   - all messages from FirstKeptID onward
//
// If no valid checkpoint exists, all messages are returned as-is.
func applyLatestCheckpoint(messages []model.Message) []model.Message {
	idx, cp := latestCheckpoint(messages)
	if cp == nil || cp.Compaction == nil || cp.Compaction.FirstKeptID == "" {
		return copyMessages(messages)
	}

	firstKeptIdx := -1
	// Resolve the checkpoint boundary by ID, not by index, so persisted/reloaded
	// histories keep working even if messages were reordered elsewhere.
	for i := range messages {
		if messages[i].ID == cp.Compaction.FirstKeptID {
			firstKeptIdx = i
			break
		}
	}

	if firstKeptIdx == -1 {
		return copyMessages(messages)
	}

	result := make([]model.Message, 0, 1+len(messages)-firstKeptIdx)
	result = append(result, messages[idx])
	// Keep everything from FirstKeptID onward, skipping a duplicate if the
	// checkpoint itself happens to fall in that range.
	for i := firstKeptIdx; i < len(messages); i++ {
		if i == idx {
			continue
		}
		result = append(result, messages[i])
	}

	return result
}

// latestCheckpoint returns the newest compaction checkpoint that has a valid boundary.
func latestCheckpoint(messages []model.Message) (int, *model.Message) {
	// Scan backwards so we find the most recent checkpoint first.
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Kind != model.MessageKindCompaction || messages[i].Compaction == nil {
			continue
		}

		if messages[i].Compaction.FirstKeptID == "" {
			continue
		}

		return i, &messages[i]
	}

	return -1, nil
}

// latestSummaryText returns the summary text from the latest checkpoint.
func latestSummaryText(messages []model.Message) string {
	_, cp := latestCheckpoint(messages)
	if cp == nil {
		return ""
	}

	return firstText(*cp)
}

// findCutIndex finds the first kept message index using a backwards token window.
//
// Algorithm:
//  1. Walk from newest to oldest accumulating estimated tokens.
//  2. Stop once keepRecentTokens is reached.
//  3. If the cut lands on a tool result, walk backwards until a non-tool-result
//     message so we don't split tool call/result context.
func findCutIndex(messages []model.Message, keepRecentTokens int) int {
	if len(messages) == 0 {
		return -1
	}

	tokens := 0
	cut := -1
	// Walk backwards to keep the most recent context under the target window.
	for i := len(messages) - 1; i >= 0; i-- {
		tokens += estimateMessageTokens(messages[i])
		if tokens >= keepRecentTokens {
			cut = i
			break
		}
	}

	if cut <= 0 {
		return cut
	}

	// Never start the kept window at a tool result.
	// This avoids dangling tool outputs without the assistant request that led to it.
	for cut > 0 && messages[cut].Kind == model.MessageKindToolResult {
		cut--
	}

	return cut
}

// estimateMessagesTokens sums approximate tokens for a message slice.
func estimateMessagesTokens(messages []model.Message) int {
	total := 0
	for _, msg := range messages {
		total += estimateMessageTokens(msg)
	}

	return total
}

// estimateMessageTokens approximates token count with chars/4, Pi-mono style.
//
// This is intentionally simple and deterministic; later versions can use
// model-aware tokenizers or provider usage metadata.
func estimateMessageTokens(msg model.Message) int {
	chars := 0
	for _, p := range msg.Content {
		if p.Type != model.ContentPartTypeText {
			continue
		}
		chars += len(p.Text)
	}

	for _, tc := range msg.ToolCallRequests {
		chars += len(tc.ToolID)
		chars += len(tc.Arguments)
	}

	tokens := chars / 4
	if chars > 0 && tokens == 0 {
		return 1
	}

	return tokens
}

// serializeMessages converts messages into a text transcript for summarization.
//
// We intentionally avoid sending these as regular conversation messages because
// text transcripts make it clearer to the summarizer model that it must summarize,
// not continue the dialogue.
func serializeMessages(messages []model.Message) string {
	parts := make([]string, 0, len(messages))
	for _, msg := range messages {
		s := serializeMessage(msg)
		if s != "" {
			parts = append(parts, s)
		}
	}

	return strings.Join(parts, "\n\n")
}

// serializeMessage renders one message in transcript format.
func serializeMessage(msg model.Message) string {
	switch msg.Kind {
	case model.MessageKindUser:
		return "[User]: " + joinText(msg.Content)
	case model.MessageKindLLM:
		items := make([]string, 0, 2)
		text := joinText(msg.Content)
		if text != "" {
			items = append(items, "[LLM]: "+text)
		}
		if len(msg.ToolCallRequests) > 0 {
			calls := make([]string, 0, len(msg.ToolCallRequests))
			for _, tc := range msg.ToolCallRequests {
				calls = append(calls, fmt.Sprintf("%s(%s)", tc.ToolID, compactJSON(tc.Arguments)))
			}
			items = append(items, "[LLM tool calls]: "+strings.Join(calls, "; "))
		}
		return strings.Join(items, "\n")
	case model.MessageKindToolResult:
		tag := "[Tool Result]"
		if msg.IsError {
			tag = "[Tool Result Error]"
		}
		return tag + ": " + joinText(msg.Content)
	case model.MessageKindCompaction:
		return "[Compaction Summary]: " + joinText(msg.Content)
	default:
		return ""
	}
}

// compactJSON normalizes JSON arguments to one-line stable representation.
func compactJSON(raw []byte) string {
	if len(raw) == 0 {
		return ""
	}

	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return string(raw)
	}

	b, err := json.Marshal(v)
	if err != nil {
		return string(raw)
	}

	return string(b)
}

// joinText joins all non-empty text content parts in a message.
func joinText(parts []model.ContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}

	return strings.Join(texts, "\n")
}

// firstText returns the first non-empty text part from a message.
func firstText(msg model.Message) string {
	for _, p := range msg.Content {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			return p.Text
		}
	}

	return ""
}

// copyMessages returns a shallow copy of the message slice.
func copyMessages(messages []model.Message) []model.Message {
	if len(messages) == 0 {
		return nil
	}

	result := make([]model.Message, len(messages))
	copy(result, messages)

	return result
}

// firstMessageID returns the first non-empty message ID from a slice.
func firstMessageID(messages []model.Message) string {
	for _, msg := range messages {
		if msg.ID != "" {
			return msg.ID
		}
	}

	return ""
}
