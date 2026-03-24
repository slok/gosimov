package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	messageutil "github.com/slok/gosimov/internal/utils/message"
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

// testToolIndex builds a tool index from mock tools by calling ID() on each.
func testToolIndex(tools ...tool.Tool) map[string]tool.Tool {
	idx := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		idx[t.ID()] = t
	}
	return idx
}

func TestRunTurn(t *testing.T) {
	tests := map[string]struct {
		mock     func(tools []*toolmock.MockTool) turnConfig
		expResp  func(t *testing.T, result *turnResult)
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

		"Simple completion should return the LLM response.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText("hello back")},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 10, OutputTokens: 5},
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hello")}},
					},
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("hello back", result.Message.Content[0].Text)
				assert.Equal(model.MessageKindLLM, result.Message.Kind)
				assert.NotEmpty(result.Message.ID)
				assert.False(result.Message.CreatedAt.IsZero())
				assert.Len(result.Messages, 1)
				assert.Equal(10, result.Usage.InputTokens)
				assert.Equal(5, result.Usage.OutputTokens)
			},
		},

		"System prompt should be forwarded to the LLM.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText(req.SystemPrompt)},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					systemPrompt: "be helpful",
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("be helpful", result.Message.Content[0].Text)
			},
		},

		"LLM provider error should propagate.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return nil, fmt.Errorf("connection refused")
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
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
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
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
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
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
								Content: []model.ContentPart{model.NewContentText("truncated")},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonMaxTokens,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("truncated", result.Message.Content[0].Text)
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
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
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
							Content: []model.ContentPart{model.NewContentText("The answer is 4")},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
								Usage:      &model.Usage{InputTokens: 20, OutputTokens: 10},
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("calculator")
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"expr":"2+2"}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("4")},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("what is 2+2")}},
					},
					toolIndex: testToolIndex(tools[0]),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("The answer is 4", result.Message.Content[0].Text)
				// Messages: LLM (tool use) + tool result + LLM (complete) = 3.
				assert.Len(result.Messages, 3)
				// Check tool result message.
				assert.Equal(model.MessageKindToolResult, result.Messages[1].Kind)
				assert.Equal("tc1", result.Messages[1].ToolCallID)
				assert.Equal("4", result.Messages[1].Content[0].Text)
				assert.False(result.Messages[1].IsError)
				assert.NotEmpty(result.Messages[1].ID)
				// Usage should be aggregated.
				assert.Equal(30, result.Usage.InputTokens)
				assert.Equal(15, result.Usage.OutputTokens)
			},
		},

		"Tool execution should use child context with timeout when configured.": {
			mock: func(tools []*toolmock.MockTool) turnConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{{
									ID:        "tc1",
									ToolID:    "slow-tool",
									Arguments: json.RawMessage(`{}`),
								}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("slow-tool")
				tools[0].On("Execute", mock.MatchedBy(func(ctx context.Context) bool {
					_, ok := ctx.Deadline()
					return ok
				}), json.RawMessage(`{}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return turnConfig{
					provider:    provider,
					messages:    []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("run")}}},
					toolIndex:   testToolIndex(tools[0]),
					toolTimeout: 2 * time.Second,
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal("done", result.Message.Content[0].Text)
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
							Content: []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("tool-a")
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"a":1}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("result-a")},
				}, nil)

				tools[1].On("ID").Return("tool-b")
				tools[1].On("Execute", mock.Anything, json.RawMessage(`{"b":2}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("result-b")},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("do both")}},
					},
					toolIndex: testToolIndex(tools[0], tools[1]),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("done", result.Message.Content[0].Text)
				// Messages: LLM (tool use) + tool result a + tool result b + LLM (complete) = 4.
				assert.Len(result.Messages, 4)
				assert.Equal("tc1", result.Messages[1].ToolCallID)
				assert.Equal("result-a", result.Messages[1].Content[0].Text)
				assert.Equal("tc2", result.Messages[2].ToolCallID)
				assert.Equal("result-b", result.Messages[2].Content[0].Text)
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
								Content: []model.ContentPart{model.NewContentText("all done")},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}
				})

				tools[0].On("ID").Return("step")
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"n":1}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("step-1-done")},
				}, nil).Once()
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"n":2}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("step-2-done")},
				}, nil).Once()

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("do steps")}},
					},
					toolIndex: testToolIndex(tools[0]),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("all done", result.Message.Content[0].Text)
				// Messages: LLM1 + tool1 + LLM2 + tool2 + LLM3 = 5.
				assert.Len(result.Messages, 5)
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
							Content: []model.ContentPart{model.NewContentText("I see the error")},
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
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("use broken tool")}},
					},
					toolIndex: testToolIndex(tools[0]),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("I see the error", result.Message.Content[0].Text)
				// Messages: LLM (tool use) + error tool result + LLM (complete) = 3.
				assert.Len(result.Messages, 3)
				assert.True(result.Messages[1].IsError)
				assert.Equal("disk full", result.Messages[1].Content[0].Text)
				assert.Equal("tc1", result.Messages[1].ToolCallID)
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
							Content: []model.ContentPart{model.NewContentText("tool was missing")},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("call missing tool")}},
					},
					// No tools provided.
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("tool was missing", result.Message.Content[0].Text)
				assert.Len(result.Messages, 3)
				assert.True(result.Messages[1].IsError)
				assert.Contains(result.Messages[1].Content[0].Text, "not found")
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
					Content: []model.ContentPart{model.NewContentText("again")},
				}, nil)

				return turnConfig{
					provider:      provider,
					messages:      []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("loop")}}},
					toolIndex:     testToolIndex(tools[0]),
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
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
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
								Content: []model.ContentPart{model.NewContentText("ok")},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("original")}},
					},
				}
			},
			expResp: func(t *testing.T, _ *turnResult) {
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
								Content: []model.ContentPart{model.NewContentText("no metadata")},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("no metadata", result.Message.Content[0].Text)
				assert.Len(result.Messages, 1)
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
									Usage:      &model.Usage{InputTokens: 100, OutputTokens: 50},
								},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
								Usage:      &model.Usage{InputTokens: 200, OutputTokens: 100},
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
					toolIndex: testToolIndex(tools[0]),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal(300, result.Usage.InputTokens)
				assert.Equal(150, result.Usage.OutputTokens)
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
							Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("saw %d messages", secondCallMsgCount))},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
					toolIndex: testToolIndex(tools[0]),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				// Second LLM call should see 3 messages: user + LLM (tool use) + tool result.
				assert.Equal("saw 3 messages", result.Message.Content[0].Text)
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
							Content: []model.ContentPart{model.NewContentText("noted")},
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
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("validate")}},
					},
					toolIndex: testToolIndex(tools[0]),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Len(result.Messages, 3)
				assert.True(result.Messages[1].IsError)
				assert.Equal("validation failed: bad input", result.Messages[1].Content[0].Text)
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
								Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("saw %d messages", len(req.Messages)))},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						// Prepend a synthetic message.
						result := make([]model.Message, 0, len(msgs)+1)
						result = append(result, model.Message{
							Kind:    model.MessageKindUser,
							Content: []model.ContentPart{model.NewContentText("injected context")},
						})
						result = append(result, msgs...)
						return result, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				// LLM should have received 2 messages (injected + original).
				assert.Equal("saw 2 messages", result.Message.Content[0].Text)
			},
		},

		"Context processor error should propagate.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewEchoProvider(),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
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
							Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("processor called %d times", processorCallCount))},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
					toolIndex: testToolIndex(tools[0]),
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						processorCallCount++
						return msgs, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				// Processor should have been called twice (once per LLM call).
				assert.Equal("processor called 2 times", result.Message.Content[0].Text)
			},
		},

		"Context processor in-place mutations should be ephemeral.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText(req.Messages[0].Content[0].Text)},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("first")}},
					},
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						msgs[0].Content[0].Text = "mutated by processor"
						return msgs, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal("mutated by processor", result.Message.Content[0].Text)
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
								Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("llm saw %d", len(req.Messages)))},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("first")}},
						{Kind: model.MessageKindLLM, Content: []model.ContentPart{model.NewContentText("reply")}},
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("second")}},
					},
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						// Only return the last message.
						return msgs[len(msgs)-1:], nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				// LLM should see only 1 message (the filtered one), not 3.
				assert.Equal("llm saw 1", result.Message.Content[0].Text)
				// But the result should still have 1 new message (the LLM response).
				assert.Len(result.Messages, 1)
			},
		},

		"Runtime context should apply latest checkpoint before LLM call.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("saw %d messages", len(req.Messages)))},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("old")}},
						{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{model.NewContentText("old reply")}},
						{ID: "c1", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("summary")}, Compaction: &model.CompactionData{FirstKeptID: "m3"}},
						{ID: "m3", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("new")}},
					},
					compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{}, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal("saw 2 messages", result.Message.Content[0].Text)
			},
		},

		"Compactor error should propagate.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewEchoProvider(),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
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
							Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("compactor called %d times", compactorCallCount))},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				})

				tools[0].On("ID").Return("t")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return turnConfig{
					provider: provider,
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}},
					},
					toolIndex: testToolIndex(tools[0]),
					compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						compactorCallCount++
						return &agentcontext.CompactResult{}, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				// Compactor should have been called twice (once per LLM call).
				assert.Equal("compactor called 2 times", result.Message.Content[0].Text)
			},
		},

		"Checkpoint context runs before context processor.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("saw %d messages", len(req.Messages)))},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("a")}},
						{ID: "m2", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("b")}},
						{ID: "c1", Kind: model.MessageKindCompaction, Content: []model.ContentPart{model.NewContentText("sum")}, Compaction: &model.CompactionData{FirstKeptID: "m3"}},
						{ID: "m3", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("c")}},
					},
					compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{}, nil
					}),
					// Processor adds 1 message to the front.
					contextProcessor: processorFunc(func(_ context.Context, msgs []model.Message) ([]model.Message, error) {
						result := make([]model.Message, 0, len(msgs)+1)
						result = append(result, model.Message{
							Kind:    model.MessageKindUser,
							Content: []model.ContentPart{model.NewContentText("injected")},
						})
						result = append(result, msgs...)
						return result, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				// Effective checkpoint context: 2 messages, then processor adds 1.
				assert.Equal("saw 3 messages", result.Message.Content[0].Text)
			},
		},

		"Noop compactor should pass through all messages.": {
			mock: func(_ []*toolmock.MockTool) turnConfig {
				return turnConfig{
					provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("saw %d messages", len(req.Messages)))},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("a")}},
						{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("b")}},
					},
					// No compactor set — defaults() will use NoopCompactor.
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)
				assert.Equal("saw 2 messages", result.Message.Content[0].Text)
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
								Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("saw %d messages", len(req.Messages)))},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
								},
							},
						}, nil
					}),
					messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("old")}},
						{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{model.NewContentText("old reply")}},
						{ID: "m3", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("new")}},
					},
					onMessages: func(_ context.Context, msgs []model.Message) error {
						persisted = append(persisted, msgs...)
						return nil
					},
					compactor: compactorFunc(func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						compactionMsg := model.Message{
							ID:      "c1",
							Kind:    model.MessageKindCompaction,
							Content: []model.ContentPart{model.NewContentText("summary")},
							Compaction: &model.CompactionData{
								FirstKeptID:  "m3",
								TokensBefore: 500,
							},
						}
						return &agentcontext.CompactResult{
							SummaryMessage: &compactionMsg,
							Usage:          model.Usage{InputTokens: 100, OutputTokens: 50},
						}, nil
					}),
				}
			},
			expResp: func(t *testing.T, result *turnResult) {
				t.Helper()
				assert := assert.New(t)

				// LLM sees summary + first kept message.
				assert.Equal("saw 2 messages", result.Message.Content[0].Text)

				// Result should include: compaction message + LLM response = 2 new messages.
				assert.Len(result.Messages, 2)
				assert.Equal(model.MessageKindCompaction, result.Messages[0].Kind)
				assert.Equal("c1", result.Messages[0].ID)
				assert.Equal(model.MessageKindLLM, result.Messages[1].Kind)

				// Compaction usage should be aggregated.
				assert.Equal(100, result.Usage.InputTokens)
				assert.GreaterOrEqual(result.Usage.OutputTokens, 50)
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

			originalMessages := messageutil.CloneMessages(config.messages)

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

			assert.Equal(originalMessages, config.messages)

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
		"Noop compactor should return nil summary message.": {
			config: compactionConfig{
				// No compactor — defaults() sets NoopCompactor.
				messages: []model.Message{
					{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hello")}},
				},
			},
			expResp: func(t *testing.T, result *agentcontext.CompactResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Nil(result.SummaryMessage)
			},
		},

		"Compactor creating a message should persist it via onMessages.": {
			config: func() compactionConfig {
				var persisted []model.Message
				compactionMsg := model.Message{
					ID:         "c1",
					Kind:       model.MessageKindCompaction,
					Content:    []model.ContentPart{model.NewContentText("summary")},
					Compaction: &model.CompactionData{FirstKeptID: "m2"},
				}
				return compactionConfig{
					messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser},
						{ID: "m2", Kind: model.MessageKindUser},
					},
					compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							SummaryMessage: &compactionMsg,
							Usage:          model.Usage{InputTokens: 100, OutputTokens: 50},
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
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(result.SummaryMessage)
				assert.Equal("c1", result.SummaryMessage.ID)
				assert.Equal(model.MessageKindCompaction, result.SummaryMessage.Kind)
				assert.Equal(100, result.Usage.InputTokens)
				assert.Equal(50, result.Usage.OutputTokens)
			},
		},

		"Compactor with no message should not call onMessages.": {
			config: compactionConfig{
				messages: []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
				},
				compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
					return &agentcontext.CompactResult{}, nil
				}),
				onMessages: func(_ context.Context, _ []model.Message) error {
					// If this is called, the test will fail because
					// notifyMessages skips nil-message results.
					return fmt.Errorf("onMessages should not be called")
				},
			},
			expResp: func(t *testing.T, result *agentcontext.CompactResult) {
				t.Helper()
				assert := assert.New(t)

				assert.Nil(result.SummaryMessage)
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
					compactor: compactorFunc(func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							SummaryMessage: &compactionMsg,
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
			assert := assert.New(t)
			require := require.New(t)

			result, err := runCompaction(context.Background(), test.config)

			if test.expErr {
				assert.Error(err)
				return
			}

			require.NoError(err)
			require.NotNil(result)

			if test.expResp != nil {
				test.expResp(t, result)
			}
		})
	}
}

func TestRunCompactionForwardsOptions(t *testing.T) {
	tests := map[string]struct {
		opts agentcontext.CompactOptions
	}{
		"Compaction options should be forwarded to the compactor.": {
			opts: agentcontext.CompactOptions{Force: true, CustomInstructions: "focus on auth"},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			var gotOpts agentcontext.CompactOptions

			_, err := runCompaction(context.Background(), compactionConfig{
				messages: []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
				opts:     test.opts,
				compactor: compactorFunc(func(_ context.Context, _ []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
					gotOpts = opts
					return &agentcontext.CompactResult{}, nil
				}),
			})

			require.NoError(err)
			assert.True(gotOpts.Force)
			assert.Equal(test.opts.CustomInstructions, gotOpts.CustomInstructions)
		})
	}
}
