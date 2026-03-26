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
//
// Each message produced during the turn (LLM responses and tool results) is
// persisted individually via the message repository as it is created.
// Session state (messages, usage) is updated before returning.
func (s *Session) runTurn(ctx context.Context, messages []model.Message, opts PromptOptions) (*turnResult, error) {
	ctx = s.ctxWithRuntimeInfo(ctx)

	systemPrompt := s.systemPrompt
	if opts.SystemPrompt != "" {
		systemPrompt = opts.SystemPrompt
	}

	maxIterations := s.maxIterations
	if opts.TurnMaxIterations > 0 {
		maxIterations = opts.TurnMaxIterations
	}

	logger := s.sessionLogger("")
	liveMessages := messages

	var (
		newMessages []model.Message
		totalUsage  model.Usage
	)

	for iteration := 0; ; iteration++ {
		logger.WithValues(gosimovlog.KV{
			"component":      "agent.turn",
			"iteration":      iteration,
			"max_iterations": maxIterations,
			"message_count":  len(liveMessages),
		}).Debugf("Turn iteration started")

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		if maxIterations > 0 && iteration >= maxIterations {
			return nil, fmt.Errorf("agent loop exceeded max iterations (%d): %w", maxIterations, pkgerrors.ErrMaxIterations)
		}

		// Compactor: may create compaction checkpoints. Runs before the context processor.
		compactResult, err := s.runCompaction(ctx, liveMessages, agentcontext.CompactOptions{})
		if err != nil {
			return nil, fmt.Errorf("running compaction: %w", err)
		}

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		// If compaction created a checkpoint, append it to turn and full history.
		if compactResult.SummaryMessage != nil {
			newMessages = append(newMessages, *compactResult.SummaryMessage)
			totalUsage = usageutil.Add(totalUsage, compactResult.Usage)
			liveMessages = append(liveMessages, *compactResult.SummaryMessage)

			liveMessages, err = sanitizeLiveCompactionHistory(liveMessages)
			if err != nil {
				return nil, fmt.Errorf("sanitizing live history after compaction: %w", err)
			}
		}

		llmMessages := liveMessages

		// Context processor: pure transform on the (already compacted) messages.
		if s.contextProcessor != nil {
			llmMessages, err = s.contextProcessor.ProcessContext(ctx, messageutil.CloneMessages(llmMessages))
			if err != nil {
				return nil, fmt.Errorf("context processing failed: %w", err)
			}
		}

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		// Build the LLM request.
		req := llm.Request{
			SystemPrompt: systemPrompt,
			SessionID:    s.session.ID,
			Messages:     llmMessages,
			Tools:        toToolDefinitions(s.tools),
			Config:       llm.RequestConfig{EnablePromptCache: !s.disablePromptCache},
		}

		// Call the LLM.
		llmStart := time.Now()
		resp, err := s.provider.Call(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("llm call failed: %w", err)
		}

		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		// Stamp the response message with ID and timestamp.
		resp.Message.ID = id.NewULID(conventions.IDPrefixMessageLLM)
		resp.Message.CreatedAt = time.Now()

		liveMessages = append(liveMessages, resp.Message)
		newMessages = append(newMessages, resp.Message)

		if resp.Message.Metadata != nil && resp.Message.Metadata.Usage != nil {
			totalUsage = usageutil.Add(totalUsage, *resp.Message.Metadata.Usage)
		}

		// Persist the LLM response.
		if err := s.messageRepo.StoreMessages(ctx, s.session.ID, []model.Message{resp.Message}); err != nil {
			return nil, fmt.Errorf("persisting llm response: %w", err)
		}

		// Decide what to do based on stop reason.
		stopReason := model.StopReasonNone
		if resp.Message.Metadata != nil {
			stopReason = resp.Message.Metadata.StopReason
		}

		logger.WithValues(gosimovlog.KV{
			"component":            "agent.turn",
			"iteration":            iteration,
			"llm_message_count":    len(llmMessages),
			"duration":             time.Since(llmStart),
			"stop_reason":          string(stopReason),
			"prompt_cache_enabled": !s.disablePromptCache,
		}).Debugf("LLM call completed")

		switch stopReason {
		case model.StopReasonComplete, model.StopReasonMaxTokens, model.StopReasonNone:
			result := &turnResult{
				Message:      resp.Message,
				Messages:     newMessages,
				LiveMessages: liveMessages,
				Usage:        totalUsage,
			}

			s.mu.Lock()
			s.messages = result.LiveMessages
			s.usage = usageutil.Add(s.usage, result.Usage)
			s.mu.Unlock()

			return result, nil

		case model.StopReasonError:
			return nil, fmt.Errorf("llm returned error stop reason: %w", pkgerrors.ErrLLMError)

		case model.StopReasonAborted:
			return nil, fmt.Errorf("llm request was aborted: %w", pkgerrors.ErrAborted)

		case model.StopReasonToolUse:
			toolResults, err := s.executeToolCalls(ctx, resp.Message.ToolCallRequests)
			if err != nil {
				return nil, fmt.Errorf("executing tool calls: %w", err)
			}
			liveMessages = append(liveMessages, toolResults...)
			newMessages = append(newMessages, toolResults...)

		default:
			return nil, fmt.Errorf("llm returned unexpected stop reason %q: %w", stopReason, pkgerrors.ErrLLMError)
		}
	}
}

// executeToolCalls runs each tool call request and returns tool result messages.
// Each tool result is persisted individually via the message repository as it is created.
func (s *Session) executeToolCalls(ctx context.Context, requests []model.ToolCallRequest) ([]model.Message, error) {
	if err := ensureTurnContextActive(ctx); err != nil {
		return nil, err
	}

	logger := s.sessionLogger("")
	results := make([]model.Message, 0, len(requests))

	for _, req := range requests {
		if err := ensureTurnContextActive(ctx); err != nil {
			return nil, err
		}

		start := time.Now()
		msg := s.executeOneToolCall(ctx, req)
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

		if err := s.messageRepo.StoreMessages(ctx, s.session.ID, []model.Message{msg}); err != nil {
			return nil, err
		}
	}

	return results, nil
}

// executeOneToolCall executes a single tool call and returns a tool result message.
//
// If the tool returns an error, err.Error() is sent to the LLM as an error tool result.
// The agent loop never aborts on tool errors — they are always fed back to the LLM.
func (s *Session) executeOneToolCall(ctx context.Context, req model.ToolCallRequest) model.Message {
	t, ok := s.toolIndex[req.ToolID]
	if !ok {
		return newToolResultMessage(req.ID, []model.ContentPart{model.NewContentText(fmt.Sprintf("tool %q not found", req.ToolID))}, true)
	}

	toolCtx := ctx
	cancel := func() {}
	if s.toolTimeout > 0 {
		toolCtx, cancel = context.WithTimeout(ctx, s.toolTimeout)
	}
	defer cancel()

	result, err := t.Execute(toolCtx, req.Arguments)
	if err != nil {
		return newToolResultMessage(req.ID, []model.ContentPart{model.NewContentText(err.Error())}, true)
	}

	return newToolResultMessage(req.ID, result.Content, false)
}

func ensureTurnContextActive(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("turn context canceled (%v): %w", err, pkgerrors.ErrAborted)
	}

	return nil
}

// turnResult is what [Session.runTurn] returns after the turn completes.
type turnResult struct {
	// Message is the final LLM response that ended the turn.
	Message model.Message
	// Messages is all new messages generated during the turn (LLM responses + tool results).
	Messages []model.Message
	// LiveMessages is the final live/effective history after the turn ends.
	LiveMessages []model.Message
	// Usage is the aggregated token usage across all LLM calls in the turn.
	Usage model.Usage
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
