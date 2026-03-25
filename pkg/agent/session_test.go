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

func withRequiredRepos(cfg agent.SessionConfig) agent.SessionConfig {
	if cfg.SessionRepository != nil && cfg.MessageRepository != nil {
		return cfg
	}

	repo := memory.NewRepository()
	if cfg.SessionRepository == nil {
		cfg.SessionRepository = repo
	}
	if cfg.MessageRepository == nil {
		cfg.MessageRepository = repo
	}

	return cfg
}

func finalLLMFromTurnResult(t *testing.T, result *agent.SessionTurnResult) model.Message {
	t.Helper()
	require := require.New(t)

	require.NotNil(result)
	require.NotEmpty(result.NewMessages)

	for i := len(result.NewMessages) - 1; i >= 0; i-- {
		if result.NewMessages[i].Kind == model.MessageKindLLM {
			return result.NewMessages[i]
		}
	}

	t.Fatalf("no LLM message in turn result")
	return model.Message{}
}

func TestNewSession(t *testing.T) {
	tests := map[string]struct {
		config   agent.SessionConfig
		expErr   bool
		expErrIs error
		assert   func(t *testing.T, s *agent.Session)
	}{
		"Valid config should create a session with a valid identity.": {
			config: agent.SessionConfig{
				Provider: fake.NewEchoProvider(),
			},
		},

		"Initial messages should preload history and usage.": {
			config: agent.SessionConfig{
				Provider: fake.NewEchoProvider(),
				Messages: []model.Message{
					{ID: "u1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hello")}},
					{
						ID:   "a1",
						Kind: model.MessageKindLLM,
						Metadata: &model.MessageMetadata{Usage: &model.Usage{
							InputTokens:  5,
							OutputTokens: 2,
						}},
					},
				},
			},
			assert: func(t *testing.T, s *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				msgs := s.Messages()
				require.Len(msgs, 2)
				assert.Equal("u1", msgs[0].ID)
				assert.Equal("a1", msgs[1].ID)

				u := s.Usage()
				assert.Equal(5, u.InputTokens)
				assert.Equal(2, u.OutputTokens)
				assert.Equal(7, u.TotalTokens)
			},
		},

		"Missing provider should return an error.": {
			config:   agent.SessionConfig{},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Duplicate tool IDs should return validation error.": {
			config: func() agent.SessionConfig {
				t1 := toolmock.NewMockTool(t)
				t2 := toolmock.NewMockTool(t)
				t1.On("ID").Return("dup-tool")
				t2.On("ID").Return("dup-tool")
				return agent.SessionConfig{
					Provider: fake.NewEchoProvider(),
					Tools:    []tool.Tool{t1, t2},
				}
			}(),
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			s, err := agent.NewSession(context.Background(), withRequiredRepos(test.config))

			if test.expErr {
				assert.Error(err)
				assert.Nil(s)
				if test.expErrIs != nil {
					assert.ErrorIs(err, test.expErrIs)
				}
			} else {
				assert.NoError(err)
				require.NotNil(s)

				// Every session must have a non-empty identity.
				sess := s.Session()
				assert.NotEmpty(sess.ID)
				assert.False(sess.CreatedAt.IsZero())

				if test.assert != nil {
					test.assert(t, s)
				}
			}
		})
	}
}

func TestNewSessionRequiresRepositories(t *testing.T) {
	tests := map[string]struct {
		config agent.SessionConfig
		err    string
	}{
		"Missing session repository should fail": {
			config: agent.SessionConfig{
				Provider:          fake.NewEchoProvider(),
				MessageRepository: memory.NewRepository(),
			},
			err: "session repository is required",
		},
		"Missing message repository should fail": {
			config: agent.SessionConfig{
				Provider:          fake.NewEchoProvider(),
				SessionRepository: memory.NewRepository(),
			},
			err: "message repository is required",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			s, err := agent.NewSession(context.Background(), test.config)
			assert.Nil(s)
			assert.Error(err)
			assert.ErrorContains(err, test.err)
			assert.ErrorIs(err, pkgerrors.ErrNotValid)
		})
	}
}

func TestNewSessionInitialMessages(t *testing.T) {
	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"Initial messages should persist and usage should aggregate.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				messages := []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
					{ID: "m2", Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 1, OutputTokens: 1}}},
				}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
					Messages:          messages,
				}))
				require.NoError(err)

				stored, err := repo.ListMessages(context.Background(), s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(stored.Items, 2)
				assert.Equal("m1", stored.Items[0].ID)
				assert.Equal("m2", stored.Items[1].ID)

				usage := s.Usage()
				assert.Equal(1, usage.InputTokens)
				assert.Equal(1, usage.OutputTokens)
				assert.Equal(2, usage.TotalTokens)
			},
		},

		"Initial messages should keep full store but sanitize session live history.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				repo := memory.NewRepository()
				messages := []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
					{ID: "m2", Kind: model.MessageKindLLM},
					{ID: "m3", Kind: model.MessageKindUser, Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 2, OutputTokens: 1}}},
					{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "m3"}},
				}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
					Messages:          messages,
				}))
				require.NoError(err)

				stored, err := repo.ListMessages(context.Background(), s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(stored.Items, 4)

				live := s.Messages()
				require.Len(live, 2)
				assert.Equal("c1", live[0].ID)
				assert.Equal("m3", live[1].ID)

				u := s.Usage()
				assert.Equal(2, u.InputTokens)
				assert.Equal(1, u.OutputTokens)
				assert.Equal(3, u.TotalTokens)
			},
		},

		"Initial messages should be copied.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				messages := []model.Message{{ID: "m1", Kind: model.MessageKindUser}}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewEchoProvider(),
					Messages: messages,
				}))
				require.NoError(err)

				messages[0].ID = "mutated"

				got := s.Messages()
				require.Len(got, 1)
				assert.Equal("m1", got[0].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			test.run(t)
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
					{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hello")}},
					{
						ID:      "m2",
						Kind:    model.MessageKindLLM,
						Content: []model.ContentPart{model.NewContentText("hi")},
						Metadata: &model.MessageMetadata{Usage: &model.Usage{
							InputTokens:  7,
							OutputTokens: 3,
						}},
					},
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

				usage := s.Usage()
				assert.Equal(7, usage.InputTokens)
				assert.Equal(3, usage.OutputTokens)
				assert.Equal(10, usage.TotalTokens)
			},
		},

		"Missing message repository should return validation error.": {
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
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Provided messages should override repository preload.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				require := require.New(t)

				repo := memory.NewRepository()
				require.NoError(repo.CreateSession(context.Background(), model.Session{ID: "s-load-override", CreatedAt: time.Now().UTC()}))
				require.NoError(repo.StoreMessages(context.Background(), "s-load-override", []model.Message{
					{ID: "repo-m1", Kind: model.MessageKindUser},
					{ID: "repo-m2", Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 20, OutputTokens: 10}}},
				}))

				return agent.LoadSessionConfig{
					SessionID:         "s-load-override",
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
					Messages: []model.Message{
						{ID: "custom-m1", Kind: model.MessageKindUser},
						{ID: "custom-m2", Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 3, OutputTokens: 2}}},
					},
				}
			},
			assert: func(t *testing.T, s *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				msgs := s.Messages()
				require.Len(msgs, 2)
				assert.Equal("custom-m1", msgs[0].ID)
				assert.Equal("custom-m2", msgs[1].ID)

				u := s.Usage()
				assert.Equal(3, u.InputTokens)
				assert.Equal(2, u.OutputTokens)
				assert.Equal(5, u.TotalTokens)
			},
		},

		"Provided empty messages should behave like nil and load from repository.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				require := require.New(t)

				repo := memory.NewRepository()
				require.NoError(repo.CreateSession(context.Background(), model.Session{ID: "s-load-empty-override", CreatedAt: time.Now().UTC()}))
				require.NoError(repo.StoreMessages(context.Background(), "s-load-empty-override", []model.Message{
					{ID: "repo-m1", Kind: model.MessageKindUser},
				}))

				return agent.LoadSessionConfig{
					SessionID:         "s-load-empty-override",
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
					Messages:          []model.Message{},
				}
			},
			assert: func(t *testing.T, s *agent.Session) {
				t.Helper()
				assert := assert.New(t)

				msgs := s.Messages()
				assert.Len(msgs, 1)
				assert.Equal("repo-m1", msgs[0].ID)
				assert.Equal(model.Usage{}, s.Usage())
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
					msg := model.Message{ID: fmt.Sprintf("m-%03d", i), Kind: model.MessageKindUser}
					if i == 1 || i == 77 {
						msg.Kind = model.MessageKindLLM
						msg.Metadata = &model.MessageMetadata{Usage: &model.Usage{
							InputTokens:  i + 1,
							OutputTokens: i + 2,
						}}
					}

					many = append(many, msg)
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

				usage := s.Usage()
				assert.Equal(80, usage.InputTokens)
				assert.Equal(82, usage.OutputTokens)
				assert.Equal(162, usage.TotalTokens)
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
				repo := memory.NewRepository()
				return agent.LoadSessionConfig{SessionID: "does-not-exist", Provider: fake.NewEchoProvider(), SessionRepository: repo, MessageRepository: repo}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotFound,
		},

		"Invalid latest checkpoint in loaded repository messages should fail.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				require := require.New(t)

				repo := memory.NewRepository()
				require.NoError(repo.CreateSession(context.Background(), model.Session{ID: "s-load-invalid-checkpoint", CreatedAt: time.Now().UTC()}))
				require.NoError(repo.StoreMessages(context.Background(), "s-load-invalid-checkpoint", []model.Message{
					{ID: "m1", Kind: model.MessageKindUser},
					{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{}},
				}))

				return agent.LoadSessionConfig{
					SessionID:         "s-load-invalid-checkpoint",
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
		},

		"Invalid latest checkpoint in provided load messages should fail.": {
			prepare: func(t *testing.T) agent.LoadSessionConfig {
				t.Helper()
				require := require.New(t)

				repo := memory.NewRepository()
				require.NoError(repo.CreateSession(context.Background(), model.Session{ID: "s-load-invalid-checkpoint-override", CreatedAt: time.Now().UTC()}))

				return agent.LoadSessionConfig{
					SessionID:         "s-load-invalid-checkpoint-override",
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
					Messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser},
						{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{}},
					},
				}
			},
			expErr:   true,
			expErrIs: pkgerrors.ErrNotValid,
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

func TestLoadSessionProvidedMessagesAreCopied(t *testing.T) {
	tests := map[string]struct {
		messages   []model.Message
		mutate     func([]model.Message)
		expFirstID string
	}{
		"Single message should be copied.": {
			messages:   []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
			mutate:     func(msgs []model.Message) { msgs[0].ID = "mutated" },
			expFirstID: "m1",
		},
		"Multiple messages should be copied.": {
			messages: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
			mutate:     func(msgs []model.Message) { msgs[0].ID = "changed" },
			expFirstID: "m1",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			repo := memory.NewRepository()
			sessionID := fmt.Sprintf("s-load-copy-%d", time.Now().UnixNano())
			require.NoError(repo.CreateSession(context.Background(), model.Session{ID: sessionID, CreatedAt: time.Now().UTC()}))

			s, err := agent.LoadSession(context.Background(), agent.LoadSessionConfig{
				SessionID:         sessionID,
				Provider:          fake.NewEchoProvider(),
				SessionRepository: repo,
				MessageRepository: repo,
				Messages:          test.messages,
			})
			require.NoError(err)

			test.mutate(test.messages)

			got := s.Messages()
			require.Len(got, len(test.messages))
			assert.Equal(test.expFirstID, got[0].ID)
		})
	}
}

func TestSessionPrompt(t *testing.T) {
	tests := map[string]struct {
		mock     func(tools []*toolmock.MockTool) agent.SessionConfig
		prompts  [][]model.ContentPart
		expResp  func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session)
		expErr   bool
		expErrIs error
	}{
		"Simple prompt should return the LLM response and accumulate messages.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
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
				}
			},
			prompts: [][]model.ContentPart{
				{model.NewContentText("hello")},
			},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("hello back", finalLLMFromTurnResult(t, results[0]).Content[0].Text)

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
								Content: []model.ContentPart{model.NewContentText(fmt.Sprintf("response %d (saw %d msgs)", callCount, len(req.Messages)))},
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
				{model.NewContentText("first")},
				{model.NewContentText("second")},
				{model.NewContentText("third")},
			},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 3)

				// First turn: LLM sees 1 message (user).
				assert.Equal("response 1 (saw 1 msgs)", finalLLMFromTurnResult(t, results[0]).Content[0].Text)
				// Second turn: LLM sees 3 messages (user + LLM + user).
				assert.Equal("response 2 (saw 3 msgs)", finalLLMFromTurnResult(t, results[1]).Content[0].Text)
				// Third turn: LLM sees 5 messages.
				assert.Equal("response 3 (saw 5 msgs)", finalLLMFromTurnResult(t, results[2]).Content[0].Text)

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
							Content:  []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				tools[0].On("ID").Return("calc")
				tools[0].On("Description").Return("calculator")
				tools[0].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("42")},
				}, nil)

				return agent.SessionConfig{
					Provider: provider,
					Tools:    []tool.Tool{tools[0]},
				}
			},
			prompts: [][]model.ContentPart{
				{model.NewContentText("calculate")},
			},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("done", finalLLMFromTurnResult(t, results[0]).Content[0].Text)

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
								Content:  []model.ContentPart{model.NewContentText("recovered")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
				}
			},
			prompts: [][]model.ContentPart{
				{model.NewContentText("hello")},
			},
			expErr: true,
			expResp: func(t *testing.T, _ []*agent.SessionTurnResult, session *agent.Session) {
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
								Content:  []model.ContentPart{model.NewContentText(text)},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
				}
			},
			prompts: [][]model.ContentPart{
				{
					model.NewContentText("what is this?"),
					model.NewContentImage([]byte("fake-png"), "image/png"),
				},
			},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, _ *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("got 2 parts", finalLLMFromTurnResult(t, results[0]).Content[0].Text)
			},
		},

		// -- Stop reason edge cases --

		"StopReasonError should return ErrLLMError.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonError},
							},
						}, nil
					}),
				}
			},
			prompts:  [][]model.ContentPart{{model.NewContentText("hi")}},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},

		"StopReasonAborted should return ErrAborted.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonAborted},
							},
						}, nil
					}),
				}
			},
			prompts:  [][]model.ContentPart{{model.NewContentText("hi")}},
			expErr:   true,
			expErrIs: pkgerrors.ErrAborted,
		},

		"Unknown stop reason should return ErrLLMError.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Metadata: &model.MessageMetadata{StopReason: model.StopReason("unexpected_reason")},
							},
						}, nil
					}),
				}
			},
			prompts:  [][]model.ContentPart{{model.NewContentText("hi")}},
			expErr:   true,
			expErrIs: pkgerrors.ErrLLMError,
		},

		"StopReasonMaxTokens should return a valid truncated result.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
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
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("hi")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, _ *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("truncated", finalLLMFromTurnResult(t, results[0]).Content[0].Text)
			},
		},

		"No metadata on LLM response should treat as complete.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText("no metadata")},
							},
						}, nil
					}),
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("hi")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("no metadata", finalLLMFromTurnResult(t, results[0]).Content[0].Text)
				assert.Len(session.Messages(), 2) // user + LLM
			},
		},

		// -- Tool execution edge cases --

		"Multiple tool calls in one response should all execute.": {
			mock: func(tools []*toolmock.MockTool) agent.SessionConfig {
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
							Kind:     model.MessageKindLLM,
							Content:  []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				tools[0].On("ID").Return("tool-a")
				tools[0].On("Description").Return("tool a")
				tools[0].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"a":1}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("result-a")},
				}, nil)

				tools[1].On("ID").Return("tool-b")
				tools[1].On("Description").Return("tool b")
				tools[1].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[1].On("Execute", mock.Anything, json.RawMessage(`{"b":2}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("result-b")},
				}, nil)

				return agent.SessionConfig{
					Provider: provider,
					Tools:    []tool.Tool{tools[0], tools[1]},
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("do both")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("done", finalLLMFromTurnResult(t, results[0]).Content[0].Text)

				// Session: user + LLM(tool use) + tool-a + tool-b + LLM(complete) = 5.
				msgs := session.Messages()
				assert.Len(msgs, 5)
				assert.Equal(model.MessageKindToolResult, msgs[2].Kind)
				assert.Equal("result-a", msgs[2].Content[0].Text)
				assert.Equal(model.MessageKindToolResult, msgs[3].Kind)
				assert.Equal("result-b", msgs[3].Content[0].Text)
			},
		},

		"Multi-round tool calls should loop until the LLM completes.": {
			mock: func(tools []*toolmock.MockTool) agent.SessionConfig {
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
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{model.NewContentText("all done")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}
				})

				tools[0].On("ID").Return("step")
				tools[0].On("Description").Return("step tool")
				tools[0].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"n":1}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("step-1-done")},
				}, nil).Once()
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{"n":2}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("step-2-done")},
				}, nil).Once()

				return agent.SessionConfig{
					Provider: provider,
					Tools:    []tool.Tool{tools[0]},
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("do steps")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("all done", finalLLMFromTurnResult(t, results[0]).Content[0].Text)

				// Session: user + LLM1 + tool1 + LLM2 + tool2 + LLM3 = 6.
				msgs := session.Messages()
				assert.Len(msgs, 6)
			},
		},

		"Tool execution error should become error result and loop continues.": {
			mock: func(tools []*toolmock.MockTool) agent.SessionConfig {
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
							Kind:     model.MessageKindLLM,
							Content:  []model.ContentPart{model.NewContentText("I see the error")},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				tools[0].On("ID").Return("broken")
				tools[0].On("Description").Return("broken tool")
				tools[0].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[0].On("Execute", mock.Anything, json.RawMessage(`{}`)).Return(nil, fmt.Errorf("disk full"))

				return agent.SessionConfig{
					Provider: provider,
					Tools:    []tool.Tool{tools[0]},
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("use broken tool")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("I see the error", finalLLMFromTurnResult(t, results[0]).Content[0].Text)

				// Session: user + LLM(tool use) + error tool result + LLM(complete) = 4.
				msgs := session.Messages()
				require.Len(msgs, 4)
				assert.True(msgs[2].IsError)
				assert.Equal("disk full", msgs[2].Content[0].Text)
			},
		},

		"Tool not found should create error result and continue.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
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
							Kind:     model.MessageKindLLM,
							Content:  []model.ContentPart{model.NewContentText("tool was missing")},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				return agent.SessionConfig{
					Provider: provider,
					// No tools provided.
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("call missing tool")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("tool was missing", finalLLMFromTurnResult(t, results[0]).Content[0].Text)

				msgs := session.Messages()
				require.Len(msgs, 4)
				assert.True(msgs[2].IsError)
				assert.Contains(msgs[2].Content[0].Text, "not found")
			},
		},

		"Tool timeout should set context deadline.": {
			mock: func(tools []*toolmock.MockTool) agent.SessionConfig {
				callCount := 0
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					callCount++
					if callCount == 1 {
						return &llm.Response{
							Message: model.Message{
								Kind: model.MessageKindLLM,
								ToolCallRequests: []model.ToolCallRequest{
									{ID: "tc1", ToolID: "slow-tool", Arguments: json.RawMessage(`{}`)},
								},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonToolUse},
							},
						}, nil
					}
					return &llm.Response{
						Message: model.Message{
							Kind:     model.MessageKindLLM,
							Content:  []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				tools[0].On("ID").Return("slow-tool")
				tools[0].On("Description").Return("slow tool")
				tools[0].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[0].On("Execute", mock.MatchedBy(func(ctx context.Context) bool {
					_, ok := ctx.Deadline()
					return ok
				}), json.RawMessage(`{}`)).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return agent.SessionConfig{
					Provider:    provider,
					Tools:       []tool.Tool{tools[0]},
					ToolTimeout: 2 * time.Second,
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("run")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, _ *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				assert.Equal("done", finalLLMFromTurnResult(t, results[0]).Content[0].Text)
			},
		},

		"Usage should be aggregated across tool call iterations within a turn.": {
			mock: func(tools []*toolmock.MockTool) agent.SessionConfig {
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
				tools[0].On("Description").Return("tool t")
				tools[0].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return agent.SessionConfig{
					Provider: provider,
					Tools:    []tool.Tool{tools[0]},
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("hi")}},
			expResp: func(t *testing.T, _ []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)

				usage := session.Usage()
				assert.Equal(300, usage.InputTokens)
				assert.Equal(150, usage.OutputTokens)
			},
		},

		// -- Within-turn compaction --

		"Compactor creating a message mid-turn should append to history.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
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
					Messages: []model.Message{
						{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("old")}},
						{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{model.NewContentText("old reply")}},
					},
					Compactor: &testCompactor{fn: func(_ context.Context, msgs []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						// Dynamically reference the last message (the new user prompt) as first kept.
						lastID := msgs[len(msgs)-1].ID
						compactionMsg := model.Message{
							ID:      "c1",
							Kind:    model.MessageKindCompaction,
							Content: []model.ContentPart{model.NewContentText("summary")},
							Compaction: &model.CompactionData{
								FirstKeptID: lastID,
							},
						}
						return &agentcontext.CompactResult{
							SummaryMessage: &compactionMsg,
							Usage:          model.Usage{InputTokens: 100, OutputTokens: 50},
						}, nil
					}},
				}
			},
			prompts: [][]model.ContentPart{
				{model.NewContentText("new")},
			},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				// LLM sees compaction summary + last message = 2 messages.
				assert.Equal("saw 2 messages", finalLLMFromTurnResult(t, results[0]).Content[0].Text)

				// Compaction usage should be aggregated.
				usage := session.Usage()
				assert.GreaterOrEqual(usage.InputTokens, 100)
				assert.GreaterOrEqual(usage.OutputTokens, 50)
			},
		},

		"Compactor should run on every iteration in the turn loop.": {
			mock: func(tools []*toolmock.MockTool) agent.SessionConfig {
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
				tools[0].On("Description").Return("tool t")
				tools[0].On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				tools[0].On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("ok")},
				}, nil)

				return agent.SessionConfig{
					Provider: provider,
					Tools:    []tool.Tool{tools[0]},
					Compactor: &testCompactor{fn: func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						compactorCallCount++
						return &agentcontext.CompactResult{}, nil
					}},
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("hi")}},
			expResp: func(t *testing.T, results []*agent.SessionTurnResult, _ *agent.Session) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				require.Len(results, 1)
				// Compactor should have been called twice (once per LLM call).
				assert.Equal("compactor called 2 times", finalLLMFromTurnResult(t, results[0]).Content[0].Text)
			},
		},

		// -- Context cancellation --

		"Context cancellation should propagate.": {
			mock: func(_ []*toolmock.MockTool) agent.SessionConfig {
				return agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return nil, context.Canceled
					}),
				}
			},
			prompts: [][]model.ContentPart{{model.NewContentText("hi")}},
			expErr:  true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			mockTools := []*toolmock.MockTool{
				toolmock.NewMockTool(t),
				toolmock.NewMockTool(t),
			}

			cfg := withRequiredRepos(test.mock(mockTools))
			session, err := agent.NewSession(context.Background(), cfg)
			require.NoError(err)

			var results []*agent.SessionTurnResult
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
				if test.expErrIs != nil {
					assert.ErrorIs(lastErr, test.expErrIs)
				}
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
		expResp func(t *testing.T, result *agent.SessionTurnResult, session *agent.Session)
		expErr  bool
	}{
		"Continue with no messages should return an error.": {
			setup: func(t *testing.T) *agent.Session {
				t.Helper()
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{Provider: fake.NewEchoProvider()}))
				require.NoError(err)

				return s
			},
			expErr: true,
		},

		"Continue with preloaded history should run a turn.": {
			setup: func(t *testing.T) *agent.Session {
				t.Helper()
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText("continued")},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 5},
								},
							},
						}, nil
					}),
					Messages: []model.Message{{
						ID:      "manual-1",
						Kind:    model.MessageKindUser,
						Content: []model.ContentPart{model.NewContentText("injected")},
					}},
				}))
				require.NoError(err)

				return s
			},
			expResp: func(t *testing.T, result *agent.SessionTurnResult, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)

				assert.Equal("continued", finalLLMFromTurnResult(t, result).Content[0].Text)
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

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{Provider: fake.NewEchoProvider()}))
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

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:    model.MessageKindLLM,
								Content: []model.ContentPart{model.NewContentText("ok")},
								Metadata: &model.MessageMetadata{
									StopReason: model.StopReasonComplete,
									Usage:      &model.Usage{InputTokens: 11, OutputTokens: 7},
								},
							},
						}, nil
					}),
				}))
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("first")}, agent.PromptOptions{})
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("second")}, agent.PromptOptions{})
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
				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						close(started)
						<-release
						return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete}}}, nil
					}),
				}))
				require.NoError(err)

				errCh := make(chan error, 1)
				go func() {
					_, err := s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("slow")}, agent.PromptOptions{})
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
				compactor := &testCompactor{fn: func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
					close(started)
					<-release
					return &agentcontext.CompactResult{}, nil
				}}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
					Messages:  []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
				}))
				require.NoError(err)

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

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return nil, fmt.Errorf("boom")
					}),
				}))
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("hi")}, agent.PromptOptions{})
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
					_, _ = session.Prompt(context.Background(), []model.ContentPart{model.NewContentText("slow")}, agent.PromptOptions{})
				}()

				// Give the first prompt time to start.
				time.Sleep(50 * time.Millisecond)

				// Second prompt should get ErrSessionBusy.
				_, err := session.Prompt(context.Background(), []model.ContentPart{model.NewContentText("fast")}, agent.PromptOptions{})
				assert.ErrorIs(err, pkgerrors.ErrSessionBusy)

				wg.Wait()
			},
		},

		"Concurrent Continue should return ErrSessionBusy.": {
			run: func(t *testing.T, session *agent.Session) {
				t.Helper()
				assert := assert.New(t)

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

			session, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
				Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					// Simulate a slow LLM call.
					time.Sleep(200 * time.Millisecond)
					return &llm.Response{
						Message: model.Message{
							Kind:    model.MessageKindLLM,
							Content: []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{
								StopReason: model.StopReasonComplete,
							},
						},
					}, nil
				}),
				Messages: []model.Message{{
					Kind:    model.MessageKindUser,
					Content: []model.ContentPart{model.NewContentText("msg")},
				}},
			}))
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

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewEchoProvider(),
					Messages: []model.Message{{ID: "1", Kind: model.MessageKindUser}},
				}))
				require.NoError(err)

				msgs := s.Messages()
				msgs[0].ID = "mutated"

				// Session's internal state should be unaffected.
				assert.Equal("1", s.Messages()[0].ID)
			},
		},

		"PromptOptions SystemPrompt should override session SystemPrompt for one call.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{model.NewContentText(req.SystemPrompt)},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SystemPrompt: "default",
				}))
				require.NoError(err)

				r1, err := s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("hi")}, agent.PromptOptions{})
				require.NoError(err)
				assert.Equal("default", finalLLMFromTurnResult(t, r1).Content[0].Text)

				r2, err := s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("hi")}, agent.PromptOptions{SystemPrompt: "override"})
				require.NoError(err)
				assert.Equal("override", finalLLMFromTurnResult(t, r2).Content[0].Text)
			},
		},

		"PromptOptions TurnMaxIterations should override session TurnMaxIterations for one call.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
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
				}))
				require.NoError(err)

				_, err = s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("hello")}, agent.PromptOptions{TurnMaxIterations: 1})
				require.Error(err)
				assert.ErrorIs(err, pkgerrors.ErrMaxIterations)
			},
		},

		"Session identity should be stable across multiple turns.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{model.NewContentText("ok")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
				}))
				require.NoError(err)

				original := s.Session()

				// Run several turns.
				for range 3 {
					_, err := s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("hi")}, agent.PromptOptions{})
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

				s1, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{Provider: fake.NewEchoProvider()}))
				require.NoError(err)

				s2, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{Provider: fake.NewEchoProvider()}))
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
								Content:  []model.ContentPart{model.NewContentText("hi back")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				sessionID = s.Session().ID

				_, err = s.Prompt(ctx, []model.ContentPart{model.NewContentText("hello")}, agent.PromptOptions{})
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
								Content:  []model.ContentPart{model.NewContentText("continued")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SessionRepository: repo,
					MessageRepository: repo,
					Messages: []model.Message{{
						ID:      "manual-1",
						Kind:    model.MessageKindUser,
						Content: []model.ContentPart{model.NewContentText("injected")},
					}},
				})
				require.NoError(err)

				_, err = s.Continue(ctx, agent.PromptOptions{})
				require.NoError(err)

				// Continue should persist only the turn result (LLM message).
				result, err := repo.ListMessages(ctx, s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 2)
				assert.Equal(model.MessageKindUser, result.Items[0].Kind)
				assert.Equal(model.MessageKindLLM, result.Items[1].Kind)
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
								Content:  []model.ContentPart{model.NewContentText("ok")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				for range 3 {
					_, err := s.Prompt(ctx, []model.ContentPart{model.NewContentText("hi")}, agent.PromptOptions{})
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
							Content:  []model.ContentPart{model.NewContentText("done")},
							Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
						},
					}, nil
				})

				mockTool := toolmock.NewMockTool(t)
				mockTool.On("ID").Return("calc")
				mockTool.On("Description").Return("calculator")
				mockTool.On("Schema").Return(json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`))
				mockTool.On("Execute", mock.Anything, mock.Anything).Return(&tool.Result{
					Content: []model.ContentPart{model.NewContentText("42")},
				}, nil)

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          provider,
					Tools:             []tool.Tool{mockTool},
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				sessionID = s.Session().ID

				_, err = s.Prompt(ctx, []model.ContentPart{model.NewContentText("calculate")}, agent.PromptOptions{})
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

		"Session should work with required repositories.": {
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
								Content:  []model.ContentPart{model.NewContentText("ok")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					SessionRepository: repo,
					MessageRepository: repo,
				})
				require.NoError(err)

				_, err = s.Prompt(ctx, []model.ContentPart{model.NewContentText("hi")}, agent.PromptOptions{})
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
					MessageRepository: repo,
				})
				require.NoError(err)

				s2, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					SessionRepository: repo,
					MessageRepository: repo,
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
						return &agentcontext.CompactResult{}, nil
					},
				}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
					Messages: []model.Message{{
						ID:      "m1",
						Kind:    model.MessageKindUser,
						Content: []model.ContentPart{model.NewContentText("hello")},
					}},
				}))
				require.NoError(err)

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

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewEchoProvider(),
					Messages: []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
					// No compactor — NoopCompactor used.
				}))
				require.NoError(err)

				result, err := s.Compact(context.Background())
				require.NoError(err)
				require.NotNil(result)

				assert.Nil(result.SummaryMessage, "NoopCompactor should not create a compaction message")
			},
		},

		"Compact creating a message should sanitize session live history.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				compactionMsg := model.Message{
					ID:         "compact-1",
					Kind:       model.MessageKindCompaction,
					Content:    []model.ContentPart{model.NewContentText("summary")},
					Compaction: &model.CompactionData{FirstKeptID: "m1"},
				}

				compactor := &testCompactor{
					fn: func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							SummaryMessage: &compactionMsg,
							Usage:          model.Usage{InputTokens: 100, OutputTokens: 50},
						}, nil
					},
				}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
					Messages: []model.Message{{
						ID:      "m1",
						Kind:    model.MessageKindUser,
						Content: []model.ContentPart{model.NewContentText("hello")},
					}},
				}))
				require.NoError(err)

				result, err := s.Compact(context.Background())
				require.NoError(err)
				require.NotNil(result.SummaryMessage)

				// Session live history should be sanitized after compaction.
				msgs := s.Messages()
				require.Len(msgs, 2)
				assert.Equal("compact-1", msgs[0].ID)
				assert.Equal(model.MessageKindCompaction, msgs[0].Kind)
				assert.Equal("m1", msgs[1].ID)

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
					Content:    []model.ContentPart{model.NewContentText("summary")},
					Compaction: &model.CompactionData{FirstKeptID: "m1"},
				}

				compactor := &testCompactor{
					fn: func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							SummaryMessage: &compactionMsg,
						}, nil
					},
				}

				s, err := agent.NewSession(ctx, agent.SessionConfig{
					Provider:          fake.NewEchoProvider(),
					Compactor:         compactor,
					SessionRepository: repo,
					MessageRepository: repo,
					Messages:          []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
				})
				require.NoError(err)

				_, err = s.Compact(ctx)
				require.NoError(err)

				// The compaction message should be in the store.
				result, err := repo.ListMessages(ctx, s.Session().ID, store.ListOpts{})
				require.NoError(err)
				require.Len(result.Items, 2)
				assert.Equal("m1", result.Items[0].ID)
				assert.Equal(model.MessageKindUser, result.Items[0].Kind)
				assert.Equal("compact-1", result.Items[1].ID)
				assert.Equal(model.MessageKindCompaction, result.Items[1].Kind)
			},
		},

		"Compact during running turn should return ErrSessionBusy.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
						// Simulate slow LLM.
						time.Sleep(200 * time.Millisecond)
						return &llm.Response{
							Message: model.Message{
								Kind:     model.MessageKindLLM,
								Content:  []model.ContentPart{model.NewContentText("done")},
								Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete},
							},
						}, nil
					}),
					Messages: []model.Message{{Kind: model.MessageKindUser, Content: []model.ContentPart{model.NewContentText("hi")}}},
				}))
				require.NoError(err)

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

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
					Messages:  []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
				}))
				require.NoError(err)

				_, err = s.Compact(context.Background())
				assert.Error(err)
				assert.Contains(err.Error(), "compaction boom")

				// Session state should be unmodified.
				msgs := s.Messages()
				assert.Len(msgs, 1)
				assert.Equal("m1", msgs[0].ID)
			},
		},

		"Invalid compaction summary boundary should fail.": {
			run: func(t *testing.T) {
				t.Helper()
				assert := assert.New(t)
				require := require.New(t)

				compactor := &testCompactor{
					fn: func(_ context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						return &agentcontext.CompactResult{
							SummaryMessage: &model.Message{
								ID:         "c1",
								Kind:       model.MessageKindCompaction,
								Compaction: &model.CompactionData{FirstKeptID: "missing"},
							},
						}, nil
					},
				}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:  fake.NewEchoProvider(),
					Compactor: compactor,
					Messages:  []model.Message{{ID: "m1", Kind: model.MessageKindUser}},
				}))
				require.NoError(err)

				_, err = s.Compact(context.Background())
				assert.Error(err)

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

func TestSessionRuntimeContextValues(t *testing.T) {
	tests := map[string]struct {
		modelInfo model.LLMModelInfo
		run       func(t *testing.T, wantInfo model.LLMModelInfo) (string, string, *model.LLMModelInfo, error)
	}{
		"Prompt should set runtime context values for provider calls.": {
			modelInfo: model.LLMModelInfo{ID: "ctx-model-prompt", ContextWindow: 1234, MaxOutputTokens: 256},
			run: func(t *testing.T, wantInfo model.LLMModelInfo) (string, string, *model.LLMModelInfo, error) {
				t.Helper()

				var gotSessionID string
				var gotInfo *model.LLMModelInfo

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider: fake.NewProviderWithModelInfo(func(ctx context.Context, _ llm.Request) (*llm.Response, error) {
						gotSessionID = agent.SessionIDFromCtx(ctx)
						gotInfo = agent.LLMModelInfoFromCtx(ctx)
						return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Metadata: &model.MessageMetadata{StopReason: model.StopReasonComplete}}}, nil
					}, wantInfo),
				}))
				if err != nil {
					return "", "", nil, err
				}

				_, err = s.Prompt(context.Background(), []model.ContentPart{model.NewContentText("hello")}, agent.PromptOptions{})
				return s.Session().ID, gotSessionID, gotInfo, err
			},
		},
		"Compact should set runtime context values for compactor calls.": {
			modelInfo: model.LLMModelInfo{ID: "ctx-model-compact", ContextWindow: 4096, MaxOutputTokens: 512},
			run: func(t *testing.T, wantInfo model.LLMModelInfo) (string, string, *model.LLMModelInfo, error) {
				t.Helper()

				var gotSessionID string
				var gotInfo *model.LLMModelInfo

				compactor := &testCompactor{
					fn: func(ctx context.Context, _ []model.Message, _ agentcontext.CompactOptions) (*agentcontext.CompactResult, error) {
						gotSessionID = agent.SessionIDFromCtx(ctx)
						gotInfo = agent.LLMModelInfoFromCtx(ctx)
						return &agentcontext.CompactResult{}, nil
					},
				}

				s, err := agent.NewSession(context.Background(), withRequiredRepos(agent.SessionConfig{
					Provider:  fake.NewProviderWithModelInfo(func(_ context.Context, _ llm.Request) (*llm.Response, error) { return nil, nil }, wantInfo),
					Compactor: compactor,
					Messages: []model.Message{{
						ID:      "m1",
						Kind:    model.MessageKindUser,
						Content: []model.ContentPart{model.NewContentText("hello")},
					}},
				}))
				if err != nil {
					return "", "", nil, err
				}
				_, err = s.Compact(context.Background())
				return s.Session().ID, gotSessionID, gotInfo, err
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			expSessionID, gotSessionID, gotInfo, err := test.run(t, test.modelInfo)
			require.NoError(err)
			assert.Equal(expSessionID, gotSessionID)
			if assert.NotNil(gotInfo) {
				assert.Equal(test.modelInfo, *gotInfo)
			}
		})
	}
}
