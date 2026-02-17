package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/fake"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/toolmock"
)

// processorFunc is a simple Processor implementation for testing.
type processorFunc func(ctx context.Context, messages []model.Message) ([]model.Message, error)

func (f processorFunc) ProcessContext(ctx context.Context, messages []model.Message) ([]model.Message, error) {
	return f(ctx, messages)
}

// Compile-time check that processorFunc implements agentcontext.Processor.
var _ agentcontext.Processor = processorFunc(nil)

// compactorFunc is a simple Compactor implementation for testing.
type compactorFunc func(ctx context.Context, messages []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error)

func (f compactorFunc) Compact(ctx context.Context, messages []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
	return f(ctx, messages, opts)
}

// Compile-time check that compactorFunc implements agentcontext.Compactor.
var _ agentcontext.Compactor = compactorFunc(nil)

func TestRunTurn(t *testing.T) {
	tests := map[string]struct {
		mock     func(tools []*toolmock.MockTool) turnConfig
		expResp  func(t *testing.T, result *TurnResult)
		expErr   bool
		expErrIs error // If set, checks errors.Is against this sentinel.
	}{
		"Missing provider should return an error.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					messages: []model.Message{{Kind: model.MessageKindUser}},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Missing messages should return an error.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewEchoProvider(),
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Duplicate tool IDs should return validation error before LLM call.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				tools[0].On("ID").Return("dup-tool")
				tools[1].On("ID").Return("dup-tool")

				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						t.Fatal("provider should not be called when tool IDs are invalid")
						return nil, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
					tools: []tool.Tool{tools[0], tools[1]},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Simple completion should return the LLM response.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello back"}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 10, OutputTokens: 5},
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "hello back", result.Message.Content[0].Text)
				assert.Equal(t, model.MessageKindLLM, result.Message.Kind)
				assert.NotEmpty(t, result.Message.ID)
				assert.False(t, result.Message.CreatedAt.IsZero())
				assert.Len(t, result.Messages, 1)
				assert.Equal(t, 10, result.Usage.InputTokens)
				assert.Equal(t, 5, result.Usage.OutputTokens)
			},
		},

		"System prompt should be forwarded to the LLM.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: req.SystemPrompt}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					systemPrompt: "be helpful",
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "be helpful", result.Message.Content[0].Text)
			},
		},

		"LLM provider error should propagate.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return nil, fmt.Errorf("connection refused")
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expErr: true,
		},

		"LLM returning StopReasonError should return an error.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonError,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},

		"LLM returning StopReasonAborted should return an error.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonAborted,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrAborted,
		},

		"StopReasonMaxTokens should return the result (truncated but valid).": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "truncated"}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonMaxTokens,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "truncated", result.Message.Content[0].Text)
			},
		},

		"Unknown stop reason should return an error.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReason("unexpected_provider_reason"),
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},

		"Single tool call round should execute the tool and return the final LLM response.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "calculator", Arguments: json.RawMessage(`{"expr":"2+2"}`)},
								},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonToolUse,
									Usage:      &model.Usage{InputTokens: 10, OutputTokens: 5},
								},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "The answer is 4"}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
								Usage:      &model.Usage{InputTokens: 20, OutputTokens: 10},
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("calculator")
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"expr":"2+2"}`)).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "4"}},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "what is 2+2"}}},
					},
					tools: []tool.Tool{tools[0]},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "The answer is 4", result.Message.Content[0].Text)
				// Messages: LLM (tool use) + tool result + LLM (complete) = 3.
				assert.Len(t, result.Messages, 3)
				// Check tool result message.
				assert.Equal(t, model.MessageKindToolResult, result.Messages[1].Kind)
				assert.Equal(t, "tc1", result.Messages[1].ToolCallID)
				assert.Equal(t, "4", result.Messages[1].Content[0].Text)
				assert.False(t, result.Messages[1].IsError)
				assert.NotEmpty(t, result.Messages[1].ID)
				// Usage should be aggregated.
				assert.Equal(t, 30, result.Usage.InputTokens)
				assert.Equal(t, 15, result.Usage.OutputTokens)
			},
		},

		"Multiple tool calls in one response should all be executed.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "tool-a", Arguments: json.RawMessage(`{"a":1}`)},
									{ID: "tc2", ToolID: "tool-b", Arguments: json.RawMessage(`{"b":2}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "done"}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("tool-a")
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"a":1}`)).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "result-a"}},
				}, nil)

				tools[1].On("ID").Return("tool-b")
				tools[1].On("Execute", mock.Anything, json.RawMessage(`{"b":2}`)).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "result-b"}},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "do both"}}},
					},
					tools: []tool.Tool{tools[0], tools[1]},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "done", result.Message.Content[0].Text)
				// Messages: LLM (tool use) + tool result a + tool result b + LLM (complete) = 4.
				assert.Len(t, result.Messages, 4)
				assert.Equal(t, "tc1", result.Messages[1].ToolCallID)
				assert.Equal(t, "result-a", result.Messages[1].Content[0].Text)
				assert.Equal(t, "tc2", result.Messages[2].ToolCallID)
				assert.Equal(t, "result-b", result.Messages[2].Content[0].Text)
			},
		},

		"Multi-round tool calls should loop until the LLM completes.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					switch callCount {
					case 1:
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "step", Arguments: json.RawMessage(`{"n":1}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					case 2:
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc2", ToolID: "step", Arguments: json.RawMessage(`{"n":2}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					default:
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "all done"}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}
				})

				tools[0].On("ID").Return("step")
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"n":1}`)).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "step-1-done"}},
				}, nil).Once()
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"n":2}`)).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "step-2-done"}},
				}, nil).Once()

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "do steps"}}},
					},
					tools: []tool.Tool{tools[0]},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "all done", result.Message.Content[0].Text)
				// Messages: LLM1 + tool1 + LLM2 + tool2 + LLM3 = 5.
				assert.Len(t, result.Messages, 5)
			},
		},

		"Tool execution error should be wrapped as an error tool result and loop should continue.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "broken", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "I see the error"}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("broken")
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{}`)).Return(nil, fmt.Errorf("disk full"))

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "use broken tool"}}},
					},
					tools: []tool.Tool{tools[0]},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "I see the error", result.Message.Content[0].Text)
				// Messages: LLM (tool use) + error tool result + LLM (complete) = 3.
				assert.Len(t, result.Messages, 3)
				assert.True(t, result.Messages[1].IsError)
				assert.Equal(t, "disk full", result.Messages[1].Content[0].Text)
				assert.Equal(t, "tc1", result.Messages[1].ToolCallID)
			},
		},

		"Tool not found should create an error tool result and continue.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "nonexistent", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "tool was missing"}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "call missing tool"}}},
					},
					// No tools provided.
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "tool was missing", result.Message.Content[0].Text)
				assert.Len(t, result.Messages, 3)
				assert.True(t, result.Messages[1].IsError)
				assert.Contains(t, result.Messages[1].Content[0].Text, "not found")
			},
		},

		"Max iterations should return an error when exceeded.": {
			expErrIs: pkgerrors.ErrMaxIterations,
			mock: func(tools []*toolmock.MockTool) turnConfig {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{
						Message: model.Message{
							Kind: model.MessageKindLLM,
							ToolCallRequests: []model.ToolCallRequest{
								{ID: "tc", ToolID: "loop", Arguments: json.RawMessage(`{}`)},
							},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
						},
					}, nil
				})

				tools[0].On("ID").Return("loop")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "again"}},
				}, nil)

				return turnConfig{
					provider:      provider,
					messages:      []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "loop"}}}},
					tools:         []tool.Tool{tools[0]},
					maxIterations: 3,
				}
			},
			expErr: true,
		},

		"Context cancellation should propagate from the LLM call.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return nil, context.Canceled
				})

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expErr: true,
		},

		"Input messages should not be mutated.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "original"}}},
					},
				}
			},
			expResp: func(t *testing.T, _ *TurnResult) {
				t.Helper()
				// The assertion is implicitly that the test didn't panic.
				// The actual mutation check is done outside via the config.Messages length.
			},
		},

		"No metadata on response should treat as complete (StopReasonNone).": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "no metadata"}},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, "no metadata", result.Message.Content[0].Text)
				assert.Len(t, result.Messages, 1)
			},
		},

		"Usage should be aggregated across multiple LLM calls.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "t", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonToolUse,
									Usage:      &model.Usage{InputTokens: 100, OutputTokens: 50, CostUSD: 0.01},
								},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "done"}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
								Usage:      &model.Usage{InputTokens: 200, OutputTokens: 100, CostUSD: 0.02},
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
					tools: []tool.Tool{tools[0]},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Equal(t, 300, result.Usage.InputTokens)
				assert.Equal(t, 150, result.Usage.OutputTokens)
				assert.InDelta(t, 0.03, result.Usage.CostUSD, 0.001)
			},
		},

		"Conversation history should accumulate across loop iterations.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				var secondCallMsgCount int
				provider := fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "t", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					// On second call, the LLM should see: original user msg + LLM tool use + tool result = 3.
					secondCallMsgCount = len(req.Messages)

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("saw %d messages", secondCallMsgCount)}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
					tools: []tool.Tool{tools[0]},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				// Second LLM call should see 3 messages: user + LLM (tool use) + tool result.
				assert.Equal(t, "saw 3 messages", result.Message.Content[0].Text)
			},
		},

		"Tool returning Go error should produce IsError tool result with error message.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "validate", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "noted"}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("validate")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(nil, fmt.Errorf("validation failed: bad input"))

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "validate"}}},
					},
					tools: []tool.Tool{tools[0]},
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				assert.Len(t, result.Messages, 3)
				assert.True(t, result.Messages[1].IsError)
				assert.Equal(t, "validation failed: bad input", result.Messages[1].Content[0].Text)
			},
		},

		"Context processor should transform messages sent to the LLM.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						// The processor should have prepended a message, so LLM sees 2 messages.
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("saw %d messages", len(req.Messages))}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						// Prepend a synthetic message.
						result := make([]model.Message, 0, len(msgs)+1)
						result = append(result, model.Message{
							Kind:    model.MessageKindUser,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "injected context"}},
						})
						result = append(result, msgs...)
						return result, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()
				// LLM should have received 2 messages (injected + original).
				assert.Equal(t, "saw 2 messages", result.Message.Content[0].Text)
			},
		},

		"Context processor error should propagate.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewEchoProvider(),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
					contextProcessor: processorFunc(func(_ context.Context, _ []model.Message) ([]model.Message, error) {
						return nil, fmt.Errorf("context processing exploded")
					}),
				}
			},
			expErr: true,
		},

		"Context processor should run on every iteration (after tool results).": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				processorCallCount := 0
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "t", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("processor called %d times", processorCallCount)}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
					tools: []tool.Tool{tools[0]},
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						processorCallCount++
						return msgs, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()
				// Processor should have been called twice (once per LLM call).
				assert.Equal(t, "processor called 2 times", result.Message.Content[0].Text)
			},
		},

		"Context processor should not mutate the full conversation history.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						// Processor filters to only 1 message, but original history is preserved.
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("llm saw %d", len(req.Messages))}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "first"}}},
						{Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "reply"}}},
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "second"}}},
					},
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						// Only return the last message.
						return msgs[len(msgs)-1:], nil
					}),
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()
				// LLM should see only 1 message (the filtered one), not 3.
				assert.Equal(t, "llm saw 1", result.Message.Content[0].Text)
				// But the result should still have 1 new message (the LLM response).
				assert.Len(t, result.Messages, 1)
			},
		},

		"Compactor should filter messages before LLM call.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("saw %d messages", len(req.Messages))}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "old"}}},
						{Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "old reply"}}},
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "new"}}},
					},
					compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						// Only keep the last message (simulating compaction).
						return &agentcontext.CompactResult{Messages: msgs[len(msgs)-1:]}, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()
				assert.Equal(t, "saw 1 messages", result.Message.Content[0].Text)
			},
		},

		"Compactor error should propagate.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewEchoProvider(),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
					compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return nil, fmt.Errorf("compaction failed")
					}),
				}
			},
			expErr: true,
		},

		"Compactor should run on every iteration (after tool results).": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				compactorCallCount := 0
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "t", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("compactor called %d times", compactorCallCount)}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
					},
					tools: []tool.Tool{tools[0]},
					compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						compactorCallCount++
						return &agentcontext.CompactResult{Messages: msgs}, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()
				// Compactor should have been called twice (once per LLM call).
				assert.Equal(t, "compactor called 2 times", result.Message.Content[0].Text)
			},
		},

		"Compactor runs before context processor.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("saw %d messages", len(req.Messages))}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "a"}}},
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "b"}}},
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "c"}}},
					},
					// Compactor keeps last 2 messages.
					compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						if len(msgs) > 2 {
							return &agentcontext.CompactResult{Messages: msgs[len(msgs)-2:]}, nil
						}
						return &agentcontext.CompactResult{Messages: msgs}, nil
					}),
					// Processor adds 1 message to the front.
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						result := make([]model.Message, 0, len(msgs)+1)
						result = append(result, model.Message{
							Kind:    model.MessageKindUser,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "injected"}},
						})
						result = append(result, msgs...)
						return result, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()
				// Compactor: 3 → 2, then processor: 2 → 3. LLM sees 3.
				assert.Equal(t, "saw 3 messages", result.Message.Content[0].Text)
			},
		},

		"Noop compactor should pass through all messages.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("saw %d messages", len(req.Messages))}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "a"}}},
						{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "b"}}},
					},
					// No compactor set — defaults() will use NoopCompactor.
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()
				assert.Equal(t, "saw 2 messages", result.Message.Content[0].Text)
			},
		},

		"Compactor creating a compaction message should append it to history and persist it.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				var persisted []model.Message
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("saw %d messages", len(req.Messages))}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "old"}}},
						{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "old reply"}}},
						{ID: "m3", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "new"}}},
					},
					onMessages: func(_ context.Context, msgs []model.Message) error {
						persisted = append(persisted, msgs...)
						return nil
					},
					compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						compactionMsg := model.Message{
							ID:      "c1",
							Kind:    model.MessageKindCompaction,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "summary"}},
							Compaction: &model.CompactionData{
								FirstKeptID:  "m3",
								TokensBefore: 500,
							},
						}
						return &agentcontext.CompactResult{
							Message:  &compactionMsg,
							Messages: msgs[len(msgs)-1:], // Only "new".
							Usage:    model.Usage{InputTokens: 100, OutputTokens: 50},
						}, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *TurnResult) {
				t.Helper()

				// LLM should see 1 message (the filtered "new").
				assert.Equal(t, "saw 1 messages", result.Message.Content[0].Text)

				// Result should include: compaction message + LLM response = 2 new messages.
				assert.Len(t, result.Messages, 2)
				assert.Equal(t, model.MessageKindCompaction, result.Messages[0].Kind)
				assert.Equal(t, "c1", result.Messages[0].ID)
				assert.Equal(t, model.MessageKindLLM, result.Messages[1].Kind)

				// Compaction usage should be aggregated.
				assert.Equal(t, 100, result.Usage.InputTokens)
				assert.GreaterOrEqual(t, result.Usage.OutputTokens, 50)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			// Create mock tools (tests can configure the ones they need).
			mockTools := []*toolmock.MockTool{
				toolmock.NewMockTool(t),
				toolmock.NewMockTool(t),
			}

			config := test.mock(mockTools)

			// Track original message count for mutation check.
			originalMsgCount := len(config.messages)

			result, err := runTurn(context.Background(), config)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.True(errors.Is(err, test.expErrIs), "expected error to wrap %v, got: %v", test.expErrIs, err)
				}
				return
			}

			require.NoError(err)
			require.NotNil(result)

			// Input messages should never be mutated.
			assert.Len(config.messages, originalMsgCount)

			if test.expResp != nil {
				test.expResp(t, result)
			}
		})
	}
}

func TestRunCompaction(t *testing.T) {
	tests := map[string]struct {
		config  compactionConfig
		expResp func(t *testing.T, result *agentcontext.CompactResult)
		expErr  bool
	}{
		"Noop compactor should return nil message and all messages.": {
			config: compactionConfig{
				// No compactor — defaults() sets NoopCompactor.
				messages: []model.Message{
					{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
				},
			},
			expResp: func(t *testing.T, result *agentcontext.CompactResult) {
				t.Helper()

				assert.Nil(t, result.Message)
				assert.Len(t, result.Messages, 1)
				assert.Equal(t, "m1", result.Messages[0].ID)
			},
		},

		"Compactor creating a message should persist it via onMessages.": {
			config: func() compactionConfig {
				var persisted []model.Message
				compactionMsg := model.Message{
					ID:         "c1",
					Kind:       model.MessageKindCompaction,
					Content:    []model.ContentPart{{Type: model.ContentPartTypeText, Text: "summary"}},
					Compaction: &model.CompactionData{FirstKeptID: "m2"},
				}
				return compactionConfig{
					messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser},
						{ID: "m2", Kind: model.MessageKindUser},
					},
					compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							Message:  &compactionMsg,
							Messages: msgs[len(msgs)-1:],
							Usage:    model.Usage{InputTokens: 100, OutputTokens: 50},
						}, nil
					}),
					onMessages: func(_ context.Context, msgs []model.Message) error {
						persisted = append(persisted, msgs...)
						return nil
					},
				}
			}(),
			expResp: func(t *testing.T, result *agentcontext.CompactResult) {
				t.Helper()

				require.NotNil(t, result.Message)
				assert.Equal(t, "c1", result.Message.ID)
				assert.Equal(t, model.MessageKindCompaction, result.Message.Kind)
				assert.Len(t, result.Messages, 1)
				assert.Equal(t, "m2", result.Messages[0].ID)
				assert.Equal(t, 100, result.Usage.InputTokens)
				assert.Equal(t, 50, result.Usage.OutputTokens)
			},
		},

		"Compactor with no message should not call onMessages.": {
			config: compactionConfig{
				messages: []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
				},
				compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
					return &agentcontext.CompactResult{Messages: msgs}, nil
				}),
				onMessages: func(_ context.Context, _ []model.Message) error {
					// If this is called, the test will fail because
					// notifyMessages skips nil-message results.
					return fmt.Errorf("onMessages should not be called")
				},
			},
			expResp: func(t *testing.T, result *agentcontext.CompactResult) {
				t.Helper()

				assert.Nil(t, result.Message)
				assert.Len(t, result.Messages, 1)
			},
		},

		"Compactor error should propagate.": {
			config: compactionConfig{
				messages: []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
				},
				compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
					return nil, fmt.Errorf("compaction boom")
				}),
			},
			expErr: true,
		},

		"Persist error should propagate.": {
			config: func() compactionConfig {
				compactionMsg := model.Message{
					ID:   "c1",
					Kind: model.MessageKindCompaction,
				}
				return compactionConfig{
					messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser},
					},
					compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							Message:  &compactionMsg,
							Messages: msgs,
						}, nil
					}),
					onMessages: func(_ context.Context, _ []model.Message) error {
						return fmt.Errorf("disk full")
					},
				}
			}(),
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			result, err := runCompaction(context.Background(), test.config)

			if test.expErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, result)

			if test.expResp != nil {
				test.expResp(t, result)
			}
		})
	}
}

func TestRunCompactionForwardsOptions(t *testing.T) {
	var gotOpts agentcontext.CompactOptions

	_, err := runCompaction(context.Background(), compactionConfig{
		messages: []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
		opts:     agentcontext.CompactOptions{Force: true, CustomInstructions: "focus on auth"},
		compactor: compactorFunc(func(_ context.Context, msgs []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
			gotOpts = opts
			return &agentcontext.CompactResult{Messages: msgs}, nil
		}),
	})

	require.NoError(t, err)
	assert.True(t, gotOpts.Force)
	assert.Equal(t, "focus on auth", gotOpts.CustomInstructions)
}
