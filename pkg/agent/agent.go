package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/slok/gosimov/internal/utils/id"
	messageutil "github.com/slok/gosimov/internal/utils/message"
	usageutil "github.com/slok/gosimov/internal/utils/usage"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/conventions"
	"github.com/slok/gosimov/pkg/llm"
	gosimovlog "github.com/slok/gosimov/pkg/log"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

func ensureTurnContextActive(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("turn context canceled (%v): %w", err, pkgerrors.ErrAborted)
	}

	return nil
}

// onMessagesFn is called each time new messages are produced during the turn.
// This enables per-message persistence — the callback is invoked once per LLM
// response and once per tool result, as they are created.
type onMessagesFn func(ctx context.Context, msgs []model.Message) error

// turnConfig configures a single [runTurn] invocation.
type turnConfig struct {
	session            model.Session
	provider           llm.Provider
	systemPrompt       string
	disablePromptCache bool
	messages           []model.Message
	tools              []tool.Tool
	toolIndex          map[string]tool.Tool
	toolTimeout        time.Duration
	maxIterations      int
	onMessages         onMessagesFn
	compactor          agentcontext.Compactor
	contextProcessor   agentcontext.Processor
	logger             gosimovlog.Logger
}

func (c *turnConfig) defaults() error {
	if c.provider == nil {
		return fmt.Errorf("provider is required: %w", pkgerrors.ErrNotValid)
	}

	if len(c.messages) == 0 {
		return fmt.Errorf("messages is required: %w", pkgerrors.ErrNotValid)
	}

	if c.compactor == nil {
		c.compactor = agentcontext.NoopCompactor{}
	}

	if c.logger == nil {
		c.logger = gosimovlog.Noop
	}

	return nil
}

// TurnResult is what [runTurn] returns after the turn completes.
type TurnResult struct {
	// Message is the final LLM response that ended the turn.
	Message model.Message
	// Messages is all new messages generated during the turn (LLM responses + tool results).
	Messages []model.Message
	// Usage is the aggregated token usage across all LLM calls in the turn.
	Usage model.Usage
}

// runTurn executes one turn of the agent loop: sends messages to the LLM,
// handles tool call requests, and loops until the LLM stops requesting tools.
//
// The loop:
//  1. Runs the compactor (may compact + filter messages based on checkpoints).
//  2. Runs the context processor (pure transform on the compacted messages).
//  3. Calls the LLM with the processed messages.
//  4. If the LLM responds with [model.StopReasonToolUse], executes the requested tools,
//     appends results to the conversation, and loops back to step 1.
//  5. If the LLM responds with [model.StopReasonComplete] or [model.StopReasonMaxTokens],
//     returns the result.
//  6. If the LLM responds with [model.StopReasonError], returns an error.
//
// Tool execution errors (Go errors from Execute) are wrapped as [model.MessageKindToolResult]
// with IsError=true and fed back to the LLM, letting it decide how to proceed.
//
// The input message elements are treated as immutable.
func runTurn(ctx context.Context, config turnConfig) (*TurnResult, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid agent turn config: %w", err)
	}

	allMessages := config.messages

	var (
		newMessages []model.Message
		totalUsage  model.Usage
	)

	for iteration := 0; ; iteration++ {
		config.logger.WithValues(gosimovlog.KV{
			"component":      "agent.turn",
			"iteration":      iteration,
			"max_iterations": config.maxIterations,
			"message_count":  len(allMessages),
		}).Debugf("Turn iteration started")

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		if config.maxIterations > 0 && iteration >= config.maxIterations {
			return nil, fmt.Errorf("agent loop exceeded max iterations (%d): %w", config.maxIterations, pkgerrors.ErrMaxIterations)
		}

		// Compactor: may create compaction checkpoints and filter messages
		// based on existing compaction markers. Runs before the context processor.
		compactResult, err := runCompaction(ctx, compactionConfig{
			compactor:  config.compactor,
			messages:   allMessages,
			onMessages: config.onMessages,
			opts:       agentcontext.CompactOptions{},
			logger: config.logger.WithValues(gosimovlog.KV{
				"component": "agent.compaction",
			}),
		})
		if err != nil {
			return nil, fmt.Errorf("running compaction: %w", err)
		}

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		allMessages = compactResult.Messages

		// If compaction created a checkpoint, append it to turn state.
		if compactResult.Message != nil {
			newMessages = append(newMessages, *compactResult.Message)
			totalUsage = addUsage(totalUsage, &model.MessageMetadata{Usage: &compactResult.Usage})

			// Compactors can return the checkpoint in two valid shapes:
			//   1. Message != nil and Messages already includes that checkpoint.
			//   2. Message != nil and Messages excludes it (caller must add it).
			// For the current turn state we need the checkpoint exactly once, so
			// detect by ID and append only when it is missing.
			checkpointInMessages := false
			for i := range allMessages {
				if allMessages[i].ID == compactResult.Message.ID {
					checkpointInMessages = true
					break
				}
			}

			if !checkpointInMessages {
				allMessages = append(allMessages, *compactResult.Message)
			}
		}

		llmMessages := compactResult.Messages

		// Context processor: pure transform on the (already compacted) messages.
		if config.contextProcessor != nil {
			llmMessages, err = config.contextProcessor.ProcessContext(ctx, messageutil.CloneMessages(llmMessages))
			if err != nil {
				return nil, fmt.Errorf("context processing failed: %w", err)
			}
		}

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		// Build the LLM request.
		req := llm.Request{
			SystemPrompt: config.systemPrompt,
			SessionID:    config.session.ID,
			Messages:     llmMessages,
			Tools:        toToolDefinitions(config.tools),
			Config:       llm.RequestConfig{EnablePromptCache: !config.disablePromptCache},
		}

		// Call the LLM.
		llmStart := time.Now()
		resp, err := config.provider.Call(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("llm call failed: %w", err)
		}

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		// Stamp the response message with ID and timestamp.
		resp.Message.ID = id.NewULID(conventions.IDPrefixMessageLLM)
		resp.Message.CreatedAt = time.Now()

		allMessages = append(allMessages, resp.Message)
		newMessages = append(newMessages, resp.Message)
		totalUsage = addUsage(totalUsage, resp.Message.Metadata)

		// Persist the LLM response.
		if err := notifyMessages(ctx, config.onMessages, resp.Message); err != nil {
			return nil, fmt.Errorf("persisting llm response: %w", err)
		}

		// Decide what to do based on stop reason.
		stopReason := model.StopReasonNone
		if resp.Message.Metadata != nil {
			stopReason = resp.Message.Metadata.StopReason
		}

		config.logger.WithValues(gosimovlog.KV{
			"component":            "agent.turn",
			"iteration":            iteration,
			"llm_message_count":    len(llmMessages),
			"duration":             time.Since(llmStart),
			"stop_reason":          string(stopReason),
			"prompt_cache_enabled": !config.disablePromptCache,
		}).Debugf("LLM call completed")

		switch stopReason {
		case model.StopReasonComplete, model.StopReasonMaxTokens, model.StopReasonNone:
			return &TurnResult{
				Message:  resp.Message,
				Messages: newMessages,
				Usage:    totalUsage,
			}, nil

		case model.StopReasonError:
			return nil, fmt.Errorf("llm returned error stop reason: %w", pkgerrors.ErrLLMError)

		case model.StopReasonAborted:
			return nil, fmt.Errorf("llm request was aborted: %w", pkgerrors.ErrAborted)

		case model.StopReasonToolUse:
			toolResults, err := executeToolCalls(ctx, resp.Message.ToolCallRequests, config.toolIndex, config.toolTimeout, config.onMessages, config.logger)
			if err != nil {
				return nil, fmt.Errorf("executing tool calls: %w", err)
			}
			allMessages = append(allMessages, toolResults...)
			newMessages = append(newMessages, toolResults...)

		default:
			return nil, fmt.Errorf("llm returned unexpected stop reason %q: %w", stopReason, pkgerrors.ErrLLMError)
		}
	}
}

func toToolDefinitions(tools []tool.Tool) []llm.ToolDefinition {
	if len(tools) == 0 {
		return nil
	}

	defs := make([]llm.ToolDefinition, len(tools))
	for i, t := range tools {
		defs[i] = llm.ToolDefinition{
			ID:          t.ID(),
			Description: t.Description(),
			Schema:      t.Schema(),
		}
	}

	return defs
}

// executeToolCalls runs each tool call request and returns tool result messages.
// Each tool result is persisted individually via onMessages as it is created.
func executeToolCalls(ctx context.Context, requests []model.ToolCallRequest, tools map[string]tool.Tool, toolTimeout time.Duration, onMessages onMessagesFn, logger gosimovlog.Logger) ([]model.Message, error) {
	if err := ensureTurnContextActive(ctx); err != nil {
		return nil, err
	}

	results := make([]model.Message, 0, len(requests))

	for _, req := range requests {
		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		start := time.Now()
		msg := executeOneToolCall(ctx, req, tools, toolTimeout)
		durationMS := time.Since(start).Milliseconds()

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		results = append(results, msg)

		logger.WithValues(gosimovlog.KV{
			"component":    "agent.turn",
			"tool_id":      req.ToolID,
			"tool_call_id": req.ID,
			"duration_ms":  durationMS,
			"is_error":     msg.IsError,
		}).Debugf("Tool call executed")

		if err := notifyMessages(ctx, onMessages, msg); err != nil {
			return nil, err
		}
	}

	return results, nil
}

// executeOneToolCall executes a single tool call and returns a tool result message.
//
// If the tool returns an error, err.Error() is sent to the LLM as an error tool result.
// The agent loop never aborts on tool errors — they are always fed back to the LLM.
func executeOneToolCall(ctx context.Context, req model.ToolCallRequest, tools map[string]tool.Tool, toolTimeout time.Duration) model.Message {
	t, ok := tools[req.ToolID]
	if !ok {
		return newToolResultMessage(req.ID, errorContent(fmt.Sprintf("tool %q not found", req.ToolID)), true)
	}

	toolCtx := ctx
	cancel := func() {}
	if toolTimeout > 0 {
		toolCtx, cancel = context.WithTimeout(ctx, toolTimeout)
	}
	defer cancel()

	result, err := t.Execute(toolCtx, req.Arguments)
	if err != nil {
		return newToolResultMessage(req.ID, errorContent(err.Error()), true)
	}

	return newToolResultMessage(req.ID, result.Content, false)
}

// newToolResultMessage creates a tool result message.
func newToolResultMessage(toolCallID string, content []model.ContentPart, isError bool) model.Message {
	return model.Message{
		ID:         id.NewULID(conventions.IDPrefixToolResult),
		Kind:       model.MessageKindToolResult,
		Content:    content,
		ToolCallID: toolCallID,
		IsError:    isError,
		CreatedAt:  time.Now(),
	}
}

// errorContent creates content parts for an error message.
func errorContent(msg string) []model.ContentPart {
	return []model.ContentPart{model.NewContentText(msg)}
}

// notifyMessages calls the onMessages callback if it is set.
func notifyMessages(ctx context.Context, fn onMessagesFn, msgs ...model.Message) error {
	if fn == nil || len(msgs) == 0 {
		return nil
	}

	return fn(ctx, msgs)
}

// compactionConfig configures a single [runCompaction] invocation.
type compactionConfig struct {
	compactor  agentcontext.Compactor
	messages   []model.Message
	onMessages onMessagesFn
	opts       agentcontext.CompactOptions
	logger     gosimovlog.Logger
}

func (c *compactionConfig) defaults() error {
	if c.compactor == nil {
		c.compactor = agentcontext.NoopCompactor{}
	}

	if c.logger == nil {
		c.logger = gosimovlog.Noop
	}

	return nil
}

// runCompaction executes one compaction pass.
//
// It delegates to the configured [Compactor] and, if a compaction checkpoint
// message is created, persists it via the onMessages callback.
//
// The caller is responsible for updating in-memory state (appending the message
// to session history, aggregating usage).
func runCompaction(ctx context.Context, config compactionConfig) (*agentcontext.CompactResult, error) {
	if err := config.defaults(); err != nil {
		return nil, fmt.Errorf("invalid compaction config: %w", err)
	}

	started := time.Now()

	result, err := config.compactor.Compact(ctx, config.messages, config.opts)
	if err != nil {
		return nil, fmt.Errorf("compaction failed: %w", err)
	}

	createdCheckpoint := result.Message != nil
	config.logger.WithValues(gosimovlog.KV{
		"duration_ms":        time.Since(started).Milliseconds(),
		"force":              config.opts.Force,
		"input_messages":     len(config.messages),
		"output_messages":    len(result.Messages),
		"created_checkpoint": createdCheckpoint,
	}).Debugf("Message compactor executed")

	if result.Message != nil {
		config.logger.WithValues(gosimovlog.KV{
			"duration_ms":         time.Since(started).Milliseconds(),
			"compacted_messages":  len(config.messages) - len(result.Messages),
			"usage_input_tokens":  result.Usage.InputTokens,
			"usage_output_tokens": result.Usage.OutputTokens,
		}).Infof("Compaction checkpoint created")

		if err := notifyMessages(ctx, config.onMessages, *result.Message); err != nil {
			return nil, fmt.Errorf("persisting compaction message: %w", err)
		}
	}

	return result, nil
}

// addUsage aggregates usage from a message's metadata into an existing total.
func addUsage(total model.Usage, metadata *model.MessageMetadata) model.Usage {
	if metadata == nil || metadata.Usage == nil {
		return total
	}

	u := metadata.Usage

	return usageutil.Add(total, *u)
}
