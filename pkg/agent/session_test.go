package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/agent"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/fake"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/store"
	"github.com/slok/gosimov/pkg/store/memory"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/toolmock"
)

// testCompactor is a simple Compactor for session-level tests.
type testCompactor struct {
	fn func(ctx context.Context, messages []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error)
}

func (c *testCompactor) Compact(ctx context.Context, messages []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
	return c.fn(ctx, messages, opts)
}

func TestNewSession(t *testing.T) {
	tests := map[string]struct {
		config agent.SessionConfig
		expErr bool
	}{
		"Valid config should create a session with a valid identity.": {
			config: agent.SessionConfig{
				Provider: fake.NewEchoProvider(),
			},
		},

		"Missing provider should return an error.": {
			config: agent.SessionConfig{},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			s, err := agent.NewSession(context.Background(), test.config)

			if test.expErr {
				assert.Error(err)
				assert.Nil(s)
			} else {
				assert.NoError(err)
				require.NotNil(s)

				// Every session must have a non-empty identity.
				sess := s.Session()
				assert.NotEmpty(sess.ID)
				assert.False(sess.CreatedAt.IsZero())
			}
		})
	}
}

func TestLoadSession(t *testing.T) {
	tests := map[string]struct {
		prepare  func(t *testing.T) agent.LoadSessionConfig
		expErr   bool
		expErrIs error
		assert   func(t *testing.T, s *agent.Session)
	}{
		"Valid config should load existing session and message history.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				require := require.New(t)

				repo := memory.NewRepository()
				sess := model.Session{ID: "s-load-1", CreatedAt: time.Now().Add(-1 * time.Hour).UTC()}
				require.NoError(repo.CreateSession(context.Background(), sess))

				msgs := []model.Message{
					{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
					{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
				}
				require.NoError(repo.StoreMessages(context.Background(), sess.ID, msgs))

				return agent.LoadSessionConfig{
					SessionID:         sess.ID,
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
				}
			},
			assert: func(t *testing.T, s *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(s)
				assert.Equal("s-load-1", s.Session().ID)
				assert.False(s.Session().CreatedAt.IsZero())

				msgs := s.Messages()
				require.Len(msgs, 2)
				assert.Equal("m1", msgs[0].ID)
				assert.Equal("hello", msgs[0].Content[0].Text)
				assert.Equal("m2", msgs[1].ID)
				assert.Equal("hi", msgs[1].Content[0].Text)
			},
		},

		"Message repository should be optional when loading existing session.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				require := require.New(t)

				repo := memory.NewRepository()
				require.NoError(repo.CreateSession(context.Background(), model.Session{ID: "s-load-no-msg-repo", CreatedAt: time.Now().UTC()}))

				return agent.LoadSessionConfig{
					SessionID:         "s-load-no-msg-repo",
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
				}
			},
			assert: func(t *testing.T, s *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(s)
				assert.Equal("s-load-no-msg-repo", s.Session().ID)
				assert.Len(s.Messages(), 0)
			},
		},

		"Loading with paginated message repository should include all pages.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				require := require.New(t)

				repo := memory.NewRepository()
				require.NoError(repo.CreateSession(context.Background(), model.Session{ID: "s-load-paginated", CreatedAt: time.Now().UTC()}))

				many := make([]model.Message, 0, 125)
				for i := 0; i < 125; i++ {
					many = append(many, model.Message{ID: fmt.Sprintf("m-%03d", i), Kind: model.MessageKindUser})
				}
				require.NoError(repo.StoreMessages(context.Background(), "s-load-paginated", many))

				return agent.LoadSessionConfig{
					SessionID:         "s-load-paginated",
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
				}
			},
			assert: func(t *testing.T, s *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)
				msgs := s.Messages()
				require.Len(msgs, 125)
				assert.Equal("m-000", msgs[0].ID)
				assert.Equal("m-124", msgs[124].ID)
			},
		},

		"Missing session id should return validation error.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				return agent.LoadSessionConfig{Provider: fake.NewEchoProvider(), SessionRepository: memory.NewRepository()}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Missing provider should return validation error.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				return agent.LoadSessionConfig{SessionID: "s1", SessionRepository: memory.NewRepository()}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Missing session repository should return validation error.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				return agent.LoadSessionConfig{SessionID: "s1", Provider: fake.NewEchoProvider()}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Missing persisted session should return not found error.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				return agent.LoadSessionConfig{SessionID: "does-not-exist", Provider: fake.NewEchoProvider(), SessionRepository: memory.NewRepository()}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotFound,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			cfg := test.prepare(t)

			s, err := agent.LoadSession(context.Background(), cfg)

			if test.expErr {
				assert.Error(err)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
				assert.Nil(s)
				return
			}

			require.NoError(err)
			require.NotNil(s)

			if test.assert != nil {
				test.assert(t, s)
			}
		})
	}
}

func TestSessionPrompt(t *testing.T) {
	tests := map[string]struct {
		mock    func(tools []*toolmock.MockTool) agent.SessionConfig
		prompts [][]model.ContentPart
		expResp func(t *testing.T, results []*agent.TurnResult, session *agent.Session)
		expErr  bool
	}{
		"Simple prompt should return the LLM response and accumulate messages.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
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
				}
			},
			prompts: [][]model.ContentPart{
				{{Type: model.ContentPartTypeText, Text: "hello"}},
			},
			expResp: func(t *testing.T, results []*agent.TurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("hello back", results[0].Message.Content[0].Text)

				// Session should have 2 messages: user + LLM.
				msgs := session.Messages()
				assert.Len(msgs, 2)
				assert.Equal(model.MessageKindUser, msgs[0].Kind)
				assert.Equal("hello", msgs[0].Content[0].Text)
				assert.NotEmpty(msgs[0].ID)
				assert.False(msgs[0].CreatedAt.IsZero())
				assert.Equal(model.MessageKindLLM, msgs[1].Kind)

				// Usage should be tracked.
				usage := session.Usage()
				assert.Equal(10, usage.InputTokens)
				assert.Equal(5, usage.OutputTokens)
			},
		},

		"Multi-turn conversation should accumulate messages and usage.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				callCount := 0
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						callCount++
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: fmt.Sprintf("response %d (saw %d msgs)", callCount, len(req.Messages))}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 10 * callCount, OutputTokens: 5 * callCount},
								},
							},
						}, nil
					}),
				}
			},
			prompts: [][]model.ContentPart{
				{{Type: model.ContentPartTypeText, Text: "first"}},
				{{Type: model.ContentPartTypeText, Text: "second"}},
				{{Type: model.ContentPartTypeText, Text: "third"}},
			},
			expResp: func(t *testing.T, results []*agent.TurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 3)

				// First turn: LLM sees 1 message (user).
				assert.Equal("response 1 (saw 1 msgs)", results[0].Message.Content[0].Text)
				// Second turn: LLM sees 3 messages (user + LLM + user).
				assert.Equal("response 2 (saw 3 msgs)", results[1].Message.Content[0].Text)
				// Third turn: LLM sees 5 messages.
				assert.Equal("response 3 (saw 5 msgs)", results[2].Message.Content[0].Text)

				// Session should have 6 messages: 3 user + 3 LLM.
				msgs := session.Messages()
				assert.Len(msgs, 6)

				// Usage should be aggregated: 10+20+30 = 60, 5+10+15 = 30.
				usage := session.Usage()
				assert.Equal(60, usage.InputTokens)
				assert.Equal(30, usage.OutputTokens)
			},
		},

		"Prompt with tool calls should accumulate all messages.": {
			mock: func(tools []*toolmock.MockTool) agent.SessionConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "calc", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:     model.MessageKindLLM,
							Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "done"}},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				tools[0].On("ID").Return("calc")
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "42"}},
				}, nil)

				return agent.SessionConfig{
					Provider: provider,
					Tools:    []tool.Tool{tools[0]},
				}
			},
			prompts: [][]model.ContentPart{
				{{Type: model.ContentPartTypeText, Text: "calculate"}},
			},
			expResp: func(t *testing.T, results []*agent.TurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("done", results[0].Message.Content[0].Text)

				// Session: user + LLM(tool use) + tool result + LLM(complete) = 4.
				msgs := session.Messages()
				assert.Len(msgs, 4)
				assert.Equal(model.MessageKindUser, msgs[0].Kind)
				assert.Equal(model.MessageKindLLM, msgs[1].Kind)
				assert.Equal(model.MessageKindToolResult, msgs[2].Kind)
				assert.Equal(model.MessageKindLLM, msgs[3].Kind)
			},
		},

		"LLM error should propagate without corrupting session state.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				callCount := 0
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						callCount++
						if callCount == 1 {
							return nil, fmt.Errorf("network error")
						}
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "recovered"}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
				}
			},
			prompts: [][]model.ContentPart{
				{{Type: model.ContentPartTypeText, Text: "hello"}},
			},
			expErr: true,
			expResp: func(t *testing.T, _ []*agent.TurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)

				// The user message was appended before the error, so it's in the history.
				msgs := session.Messages()
				assert.Len(msgs, 1)
				assert.Equal(model.MessageKindUser, msgs[0].Kind)
			},
		},

		"Prompt with image content should work.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						// Verify the image was received.
						lastMsg := req.Messages[len(req.Messages)-1]
						text := fmt.Sprintf("got %d parts", len(lastMsg.Content))
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: text}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
				}
			},
			prompts: [][]model.ContentPart{
				{
					{Type: model.ContentPartTypeText, Text: "what is this?"},
					{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: []byte("fake-png"), MimeType: "image/png"}},
				},
			},
			expResp: func(t *testing.T, results []*agent.TurnResult, _ *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("got 2 parts", results[0].Message.Content[0].Text)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			mockTools := []*toolmock.MockTool{
				toolmock.NewMockTool(t),
			}

			cfg := test.mock(mockTools)
			session, err := agent.NewSession(context.Background(), cfg)
			require.NoError(err)

			var results []*agent.TurnResult
			var lastErr error
			for _, content := range test.prompts {
				result, promptErr := session.Prompt(context.Background(), content, agent.PromptOptions{})
				if promptErr != nil {
					lastErr = promptErr
					break
				}
				results = append(results, result)
			}

			if test.expErr {
				assert.Error(lastErr)
			} else {
				assert.NoError(lastErr)
			}

			if test.expResp != nil {
				test.expResp(t, results, session)
			}
		})
	}
}

func TestSessionContinue(t *testing.T) {
	tests := map[string]struct {
		setup   func(t *testing.T) *agent.Session
		expResp func(t *testing.T, result *agent.TurnResult, session *agent.Session)
		expErr  bool
	}{
		"Continue with no messages should return an error.": {
			setup: func(t *testing.T) *agent.Session {
				t.Helper()
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{Provider: fake.NewEchoProvider()})
				require.NoError(err)

				return s
			},
			expErr: true,
		},

		"Continue after AppendMessage should run a turn.": {
			setup: func(t *testing.T) *agent.Session {
				t.Helper()
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "continued"}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 5},
								},
							},
						}, nil
					}),
				})
				require.NoError(err)

				s.AppendMessage(model.Message{
					ID:      "manual-1",
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "injected"}},
				})

				return s
			},
			expResp: func(t *testing.T, result *agent.TurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("continued", result.Message.Content[0].Text)
				// Messages: injected user + LLM response = 2.
				msgs := session.Messages()
				assert.Len(msgs, 2)
				assert.Equal("manual-1", msgs[0].ID)
				assert.Equal(5, session.Usage().InputTokens)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			session := test.setup(t)

			result, err := session.Continue(context.Background(), agent.PromptOptions{})

			if test.expErr {
				assert.Error(err)
				return
			}

			require.NoError(err)
			require.NotNil(result)

			if test.expResp != nil {
				test.expResp(t, result, session)
			}
		})
	}
}

func TestSessionState(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"New session should report idle state.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{Provider: fake.NewEchoProvider()})
				require.NoError(err)

				state := s.State()
				assert.False(state.Running)
				assert.Equal(agent.SessionOperationNone, state.Operation)
				assert.Zero(state.Turn)
				assert.Zero(state.MessageCount)
				assert.Zero(state.Usage.InputTokens)
				assert.Equal(s.Session().ID, state.Session.ID)
			},
		},

		"State should include turn count, message count and usage.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 11, OutputTokens: 7},
								},
							},
						}, nil
					}),
				})
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "first"}}, agent.PromptOptions{})
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "second"}}, agent.PromptOptions{})
				require.NoError(err)

				state := s.State()
				assert.False(state.Running)
				assert.Equal(agent.SessionOperationNone, state.Operation)
				assert.Equal(2, state.Turn)
				assert.Equal(4, state.MessageCount)
				assert.Equal(22, state.Usage.InputTokens)
				assert.Equal(14, state.Usage.OutputTokens)
			},
		},

		"State should report prompt operation while running.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				started := make(chan struct{})
				release := make(chan struct{})
				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						close(started)
						<-release
						return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete}}}, nil
					}),
				})
				require.NoError(err)

				errCh := make(chan error, 1)
				go func() {
					_, err := s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "slow"}}, agent.PromptOptions{})
					errCh <- err
				}()

				<-started
				state := s.State()
				assert.True(state.Running)
				assert.Equal(agent.SessionOperationPrompt, state.Operation)
				assert.Equal(1, state.MessageCount)
				assert.Equal(1, state.Turn)

				close(release)
				require.NoError(<-errCh)

				state = s.State()
				assert.False(state.Running)
				assert.Equal(agent.SessionOperationNone, state.Operation)
			},
		},

		"State should report compact operation while running.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				started := make(chan struct{})
				release := make(chan struct{})
				compactor := &testCompactor{fn: func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
					close(started)
					<-release
					return &agentcontext.CompactResult{Messages: msgs}, nil
				}}

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{Provider: fake.NewEchoProvider(), Compactor: compactor})
				require.NoError(err)
				s.AppendMessage(model.Message{ID: "m1", Kind: model.MessageKindUser})

				errCh := make(chan error, 1)
				go func() {
					_, err := s.Compact(context.Background())
					errCh <- err
				}()

				<-started
				state := s.State()
				assert.True(state.Running)
				assert.Equal(agent.SessionOperationCompact, state.Operation)
				assert.Equal(1, state.MessageCount)
				assert.Equal(1, state.Turn)

				close(release)
				require.NoError(<-errCh)

				state = s.State()
				assert.False(state.Running)
				assert.Equal(agent.SessionOperationNone, state.Operation)
			},
		},

		"State should clear operation after failed run.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return nil, fmt.Errorf("boom")
					}),
				})
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}, agent.PromptOptions{})
				require.Error(err)

				state := s.State()
				assert.False(state.Running)
				assert.Equal(agent.SessionOperationNone, state.Operation)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

func TestSessionConcurrency(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T, session *agent.Session)
	}{
		"Concurrent Prompt should return ErrSessionBusy.": {
			run: func(t *testing.T, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)

				// Start a slow prompt.
				var wg sync.WaitGroup
				wg.Add(1)
				go func() {
					defer wg.Done()
					_, _ = session.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "slow"}}, agent.PromptOptions{})
				}()

				// Give the first prompt time to start.
				time.Sleep(50 * time.Millisecond)

				// Second prompt should get ErrSessionBusy.
				_, err := session.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "fast"}}, agent.PromptOptions{})
				assert.ErrorIs(err, pkgerrors.ErrSessionBusy)

				wg.Wait()
			},
		},

		"Concurrent Continue should return ErrSessionBusy.": {
			run: func(t *testing.T, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)

				// Inject a message so Continue has something to work with.
				session.AppendMessage(model.Message{
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "msg"}},
				})

				// Start a slow continue.
				var wg sync.WaitGroup
				wg.Add(1)
				go func() {
					defer wg.Done()
					_, _ = session.Continue(context.Background(), agent.PromptOptions{})
				}()

				// Give the first call time to start.
				time.Sleep(50 * time.Millisecond)

				// Second call should get ErrSessionBusy.
				_, err := session.Continue(context.Background(), agent.PromptOptions{})
				assert.ErrorIs(err, pkgerrors.ErrSessionBusy)

				wg.Wait()
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			session, err := agent.NewSession(context.Background(), agent.SessionConfig{
				Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					// Simulate a slow LLM call.
					time.Sleep(200 * time.Millisecond)
					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "done"}},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				}),
			})
			require.NoError(err)

			test.run(t, session)
		})
	}
}

func TestSessionMutators(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Messages should return a copy that doesn't affect the session.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{Provider: fake.NewEchoProvider()})
				require.NoError(err)

				s.AppendMessage(model.Message{ID: "1", Kind: model.MessageKindUser})

				msgs := s.Messages()
				msgs[0].ID = "mutated"

				// Session's internal state should be unaffected.
				assert.Equal("1", s.Messages()[0].ID)
			},
		},

		"ReplaceMessages should replace history and not retain a reference.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{Provider: fake.NewEchoProvider()})
				require.NoError(err)

				s.AppendMessage(model.Message{ID: "old", Kind: model.MessageKindUser})

				replacement := []model.Message{{ID: "new", Kind: model.MessageKindUser}}
				s.ReplaceMessages(replacement)

				// Mutating the input should not affect the session.
				replacement[0].ID = "mutated"

				msgs := s.Messages()
				require.Len(msgs, 1)
				assert.Equal("new", msgs[0].ID)
			},
		},

		"Reset should clear messages and usage but preserve session identity.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 10},
								},
							},
						}, nil
					}),
				})
				require.NoError(err)

				sessionBefore := s.Session()

				_, err = s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}, agent.PromptOptions{})
				require.NoError(err)
				assert.NotEmpty(s.Messages())
				assert.NotZero(s.Usage().InputTokens)

				s.Reset()

				assert.Empty(s.Messages())
				assert.Zero(s.Usage().InputTokens)

				// Session identity must survive reset.
				sessionAfter := s.Session()
				assert.Equal(sessionBefore.ID, sessionAfter.ID)
				assert.Equal(sessionBefore.CreatedAt, sessionAfter.CreatedAt)
			},
		},

		"PromptOptions SystemPrompt should override session SystemPrompt for one call.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: req.SystemPrompt}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SystemPrompt: "default",
				})
				require.NoError(err)

				r1, err := s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}, agent.PromptOptions{})
				require.NoError(err)
				assert.Equal("default", r1.Message.Content[0].Text)

				r2, err := s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}, agent.PromptOptions{SystemPrompt: "override"})
				require.NoError(err)
				assert.Equal("override", r2.Message.Content[0].Text)
			},
		},

		"PromptOptions TurnMaxIterations should override session TurnMaxIterations for one call.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{Message: model.Message{
							Kind: model.MessageKindLLM,
							ToolCallRequests: []model.ToolCallRequest{{
								ID:        "tool-call-id",
								ToolID:    "missing",
								Arguments: []byte(`{}`),
							}},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
						}}, nil
					}),
					TurnMaxIterations: 2,
				})
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}, agent.PromptOptions{TurnMaxIterations: 1})
				require.Error(err)
				assert.ErrorIs(err, pkgerrors.ErrMaxIterations)
			},
		},

		"Session identity should be stable across multiple turns.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
				})
				require.NoError(err)

				original := s.Session()

				// Run several turns.
				for range 3 {
					_, err := s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}, agent.PromptOptions{})
					require.NoError(err)
				}

				// Identity must not change.
				after := s.Session()
				assert.Equal(original.ID, after.ID)
				assert.Equal(original.CreatedAt, after.CreatedAt)
			},
		},

		"Two sessions should have different IDs.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s1, err := agent.NewSession(context.Background(), agent.SessionConfig{Provider: fake.NewEchoProvider()})
				require.NoError(err)

				s2, err := agent.NewSession(context.Background(), agent.SessionConfig{Provider: fake.NewEchoProvider()})
				require.NoError(err)

				assert.NotEqual(s1.Session().ID, s2.Session().ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

func TestSessionPersistence(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"NewSession with repos should persist the session.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				ctx := context.Background()

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				got, err := repo.GetSession(ctx, s.Session().ID)
				require.NoError(err)
				assert.Equal(s.Session().ID, got.ID)
			},
		},

		"Prompt should persist user message eagerly and LLM response via callback.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				ctx := context.Background()

				// Track what's in the store at the time the LLM is called.
				var msgsAtLLMCall []model.Message
				var sessionID string

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						// At this point the user message should already be persisted.
						result, err := repo.ListMessages(ctx, sessionID, store.ListOpts{})
						if err == nil {
							msgsAtLLMCall = result.Items
						}

						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi back"}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				sessionID = s.Session().ID

				_, err = s.Prompt(ctx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}, agent.PromptOptions{})
				require.NoError(err)

				// User message was persisted before the LLM call.
				require.Len(msgsAtLLMCall, 1)
				assert.Equal(model.MessageKindUser, msgsAtLLMCall[0].Kind)
				assert.Equal("hello", msgsAtLLMCall[0].Content[0].Text)

				// After the turn, both messages are in the store.
				result, err := repo.ListMessages(ctx, s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 2)
				assert.Equal(model.MessageKindUser, result.Items[0].Kind)
				assert.Equal(model.MessageKindLLM, result.Items[1].Kind)
				assert.Equal("hi back", result.Items[1].Content[0].Text)
			},
		},

		"Continue should persist only turn messages (no extra user message).": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				ctx := context.Background()

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "continued"}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				s.AppendMessage(model.Message{
					ID:      "manual-1",
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "injected"}},
				})

				_, err = s.Continue(ctx, agent.PromptOptions{})
				require.NoError(err)

				// Continue should only persist the turn result (LLM message), not the manually appended one.
				result, err := repo.ListMessages(ctx, s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 1)
				assert.Equal(model.MessageKindLLM, result.Items[0].Kind)
			},
		},

		"Multi-turn should accumulate persisted messages across turns.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				ctx := context.Background()

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				for range 3 {
					_, err := s.Prompt(ctx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}, agent.PromptOptions{})
					require.NoError(err)
				}

				// 3 turns × (user + LLM) = 6 messages.
				result, err := repo.ListMessages(ctx, s.Session().ID, store.ListOpts{})
				require.NoError(err)
				assert.Len(result.Items, 6)
			},
		},

		"Tool use turn should persist each message individually as produced.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				ctx := context.Background()

				// Track store state at each LLM call to verify incremental persistence.
				var msgsAtFirstLLMCall, msgsAtSecondLLMCall []model.Message
				var sessionID string

				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++

					result, err := repo.ListMessages(ctx, sessionID, store.ListOpts{})
					if err == nil {
						if callCount == 1 {
							msgsAtFirstLLMCall = result.Items
						} else {
							msgsAtSecondLLMCall = result.Items
						}
					}

					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "calc", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}

					return &llm.Response{
						Message: model.Message{
							Kind:     model.MessageKindLLM,
							Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "done"}},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				mockTool := toolmock.NewMockTool(t)
				mockTool.On("ID").Return("calc")
				mockTool.On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "42"}},
				}, nil)

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          provider,
					Tools:             []tool.Tool{mockTool},
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				sessionID = s.Session().ID

				_, err = s.Prompt(ctx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: "calculate"}}, agent.PromptOptions{})
				require.NoError(err)

				// At first LLM call: only user message persisted.
				require.Len(msgsAtFirstLLMCall, 1)
				assert.Equal(model.MessageKindUser, msgsAtFirstLLMCall[0].Kind)

				// At second LLM call: user + LLM(tool use) + tool result = 3 persisted.
				require.Len(msgsAtSecondLLMCall, 3)
				assert.Equal(model.MessageKindUser, msgsAtSecondLLMCall[0].Kind)
				assert.Equal(model.MessageKindLLM, msgsAtSecondLLMCall[1].Kind)
				assert.Equal(model.MessageKindToolResult, msgsAtSecondLLMCall[2].Kind)

				// After turn: all 4 messages persisted.
				result, err := repo.ListMessages(ctx, s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 4)
				assert.Equal(model.MessageKindUser, result.Items[0].Kind)
				assert.Equal(model.MessageKindLLM, result.Items[1].Kind)
				assert.Equal(model.MessageKindToolResult, result.Items[2].Kind)
				assert.Equal(model.MessageKindLLM, result.Items[3].Kind)
			},
		},

		"Without repos, session should work as before (no persistence).": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				ctx := context.Background()

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "ok"}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					// No repos — purely in-memory.
				})
				require.NoError(err)

				_, err = s.Prompt(ctx, []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}, agent.PromptOptions{})
				require.NoError(err)

				// Session should still work.
				msgs := s.Messages()
				assert.Len(msgs, 2)
			},
		},

		"Session should be listed after creation.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				ctx := context.Background()

				s1, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
				})
				require.NoError(err)

				s2, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
				})
				require.NoError(err)

				result, err := repo.ListSessions(ctx, store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 2)

				// Newest first.
				assert.Equal(s2.Session().ID, result.Items[0].ID)
				assert.Equal(s1.Session().ID, result.Items[1].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

func TestSessionCompact(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Compact should call compactor with Force=true.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				var gotOpts agentcontext.CompactOptions
				var gotMsgs []model.Message
				compactor := &testCompactor{
					fn: func(_ context.Context, msgs []model.Message, opts agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						gotOpts = opts
						gotMsgs = msgs
						return &agentcontext.CompactResult{Messages: msgs}, nil
					},
				}

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
				})
				require.NoError(err)

				s.AppendMessage(model.Message{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}})

				result, err := s.Compact(context.Background())
				require.NoError(err)
				require.NotNil(result)

				assert.True(gotOpts.Force, "Force should be true")
				assert.Len(gotMsgs, 1)
				assert.Equal("m1", gotMsgs[0].ID)
			},
		},

		"Compact with no compactor (noop) should return nil message.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewEchoProvider(),
					// No compactor — NoopCompactor used.
				})
				require.NoError(err)

				s.AppendMessage(model.Message{ID: "m1", Kind: model.MessageKindUser})

				result, err := s.Compact(context.Background())
				require.NoError(err)
				require.NotNil(result)

				assert.Nil(result.Message, "NoopCompactor should not create a compaction message")
				assert.Len(result.Messages, 1)
			},
		},

		"Compact creating a message should append it to session history.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				compactionMsg := model.Message{
					ID:         "compact-1",
					Kind:       model.MessageKindCompaction,
					Content:    []model.ContentPart{{Type: model.ContentPartTypeText, Text: "summary"}},
					Compaction: &model.CompactionData{FirstKeptID: "m1"},
				}

				compactor := &testCompactor{
					fn: func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							Message:  &compactionMsg,
							Messages: msgs,
							Usage:    model.Usage{InputTokens: 100, OutputTokens: 50},
						}, nil
					},
				}

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
				})
				require.NoError(err)

				s.AppendMessage(model.Message{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}})

				result, err := s.Compact(context.Background())
				require.NoError(err)
				require.NotNil(result.Message)

				// The compaction message should be appended to session history.
				msgs := s.Messages()
				require.Len(msgs, 2)
				assert.Equal("m1", msgs[0].ID)
				assert.Equal("compact-1", msgs[1].ID)
				assert.Equal(model.MessageKindCompaction, msgs[1].Kind)

				// Usage should be aggregated.
				usage := s.Usage()
				assert.Equal(100, usage.InputTokens)
				assert.Equal(50, usage.OutputTokens)
			},
		},

		"Compact should persist the compaction message when repo is set.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				ctx := context.Background()

				compactionMsg := model.Message{
					ID:         "compact-1",
					Kind:       model.MessageKindCompaction,
					Content:    []model.ContentPart{{Type: model.ContentPartTypeText, Text: "summary"}},
					Compaction: &model.CompactionData{FirstKeptID: "m1"},
				}

				compactor := &testCompactor{
					fn: func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							Message:  &compactionMsg,
							Messages: msgs,
						}, nil
					},
				}

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					Compactor:         compactor,
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				s.AppendMessage(model.Message{ID: "m1", Kind: model.MessageKindUser})

				_, err = s.Compact(ctx)
				require.NoError(err)

				// The compaction message should be in the store.
				result, err := repo.ListMessages(ctx, s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 1)
				assert.Equal("compact-1", result.Items[0].ID)
				assert.Equal(model.MessageKindCompaction, result.Items[0].Kind)
			},
		},

		"Compact during running turn should return ErrSessionBusy.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						// Simulate slow LLM.
						time.Sleep(200 * time.Millisecond)
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{{Type: model.ContentPartTypeText, Text: "done"}},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
				})
				require.NoError(err)

				s.AppendMessage(model.Message{Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}})

				// Start a turn.
				var wg sync.WaitGroup
				wg.Add(1)
				go func() {
					defer wg.Done()
					_, _ = s.Continue(context.Background(), agent.PromptOptions{})
				}()

				// Give the turn time to start.
				time.Sleep(50 * time.Millisecond)

				// Compact should return ErrSessionBusy.
				_, err = s.Compact(context.Background())
				assert.ErrorIs(err, pkgerrors.ErrSessionBusy)

				wg.Wait()
			},
		},

		"Compactor error should propagate.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				compactor := &testCompactor{
					fn: func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return nil, fmt.Errorf("compaction boom")
					},
				}

				s, err := agent.NewSession(context.Background(), agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
				})
				require.NoError(err)

				s.AppendMessage(model.Message{ID: "m1", Kind: model.MessageKindUser})

				_, err = s.Compact(context.Background())
				assert.Error(err)
				assert.Contains(err.Error(), "compaction boom")

				// Session state should be unmodified.
				msgs := s.Messages()
				assert.Len(msgs, 1)
				assert.Equal("m1", msgs[0].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
		})
	}
}

func TestSessionPromptSetsRuntimeContextValues(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	wantInfo := model.LLMModelInfo{ID: "ctx-model", ContextWindow: 1234, MaxOutputTokens: 256}

	var gotSessionID string
	var gotInfo *model.LLMModelInfo

	s, err := agent.NewSession(context.Background(), agent.SessionConfig{
		Provider: fake.NewProviderWithModelInfo(func(ctx context.Context, _ llm.Request) (*llm.Response, error) {
			gotSessionID = agent.SessionIDFromCtx(ctx)
			gotInfo = agent.LLMModelInfoFromCtx(ctx)

			return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete}}}, nil
		}, wantInfo),
	})
	require.NoError(err)

	_, err = s.Prompt(context.Background(), []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}, agent.PromptOptions{})
	require.NoError(err)

	assert.Equal(s.Session().ID, gotSessionID)
	if assert.NotNil(gotInfo) {
		assert.Equal(wantInfo, *gotInfo)
	}
}

func TestSessionCompactSetsRuntimeContextValues(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	wantInfo := model.LLMModelInfo{ID: "ctx-model", ContextWindow: 4096, MaxOutputTokens: 512}

	var gotSessionID string
	var gotInfo *model.LLMModelInfo

	compactor := &testCompactor{
		fn: func(ctx context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
			gotSessionID = agent.SessionIDFromCtx(ctx)
			gotInfo = agent.LLMModelInfoFromCtx(ctx)
			return &agentcontext.CompactResult{Messages: msgs}, nil
		},
	}

	s, err := agent.NewSession(context.Background(), agent.SessionConfig{
		Provider:  fake.NewProviderWithModelInfo(func(_ context.Context, _ llm.Request) (*llm.Response, error) { return nil, nil }, wantInfo),
		Compactor: compactor,
	})
	require.NoError(err)

	s.AppendMessage(model.Message{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}})

	_, err = s.Compact(context.Background())
	require.NoError(err)

	assert.Equal(s.Session().ID, gotSessionID)
	if assert.NotNil(gotInfo) {
		assert.Equal(wantInfo, *gotInfo)
	}
}
