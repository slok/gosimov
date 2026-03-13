package simple_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/slok/gosimov/pkg/agent"
	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/agent/context/simple"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/fake"
	"github.com/slok/gosimov/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const defaultMaxSummaryTokens = 13107

func TestNew(t *testing.T) {
	tests := map[string]struct {
		cfg    simple.Config
		expErr bool
	}{
		"Missing provider should fail.": {
			cfg:    simple.Config{},
			expErr: true,
		},
		"Valid config should create compactor.": {
			cfg: simple.Config{Provider: fake.NewEchoProvider()},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			c, err := simple.New(test.cfg)
			if test.expErr {
				require.Error(err)
				return
			}

			require.NoError(err)
			require.NotNil(c)
		})
	}
}

func TestCompactorCompact(t *testing.T) {
	tests := map[string]struct {
		mock   func() agentcontext.Compactor
		ctx    context.Context
		msgs   []model.Message
		opts   agentcontext.CompactOptions
		expErr bool
		assert func(t *testing.T, got *agentcontext.CompactResult)
	}{
		"Force false should pass through without checkpoint.": {
			mock: func() agentcontext.Compactor {
				c, _ := simple.New(simple.Config{Provider: fake.NewEchoProvider()})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hello"}}},
				{ID: "m2", Kind: model.MessageKindLLM, Content: []model.ContentPart{{Type: model.ContentPartTypeText, Text: "hi"}}},
			},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				assert.Nil(got.Message)
				require.Len(got.Messages, 2)
				assert.Equal("m1", got.Messages[0].ID)
				assert.Equal("m2", got.Messages[1].ID)
			},
		},
		"Force false above threshold should compact automatically.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{
						Kind:     model.MessageKindLLM,
						Content:  textContent("auto summary"),
						Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 11, OutputTokens: 7}},
					}}, nil
				})
				c, _ := simple.New(simple.Config{
					Provider:         provider,
					ReserveTokens:    2,
					KeepRecentTokens: 2,
				})
				return c
			},
			ctx: agent.CtxWithLLMModelInfo(context.Background(), &model.LLMModelInfo{ContextWindow: 6}),
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("12345678")},
				{ID: "m3", Kind: model.MessageKindUser, Content: textContent("1234")},
			},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(got.Message)
				assert.Equal(model.MessageKindCompaction, got.Message.Kind)
				assert.Equal("m2", got.Message.Compaction.FirstKeptID)
				require.Len(got.Messages, 3)
				assert.Equal(11, got.Usage.InputTokens)
				assert.Equal(7, got.Usage.OutputTokens)
			},
		},
		"Force false should use context model info as context window fallback.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("ctx summary")}}, nil
				})
				c, _ := simple.New(simple.Config{
					Provider:         provider,
					ReserveTokens:    2,
					KeepRecentTokens: 2,
				})
				return c
			},
			ctx: agent.CtxWithLLMModelInfo(context.Background(), &model.LLMModelInfo{ContextWindow: 6}),
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("12345678")},
				{ID: "m3", Kind: model.MessageKindUser, Content: textContent("1234")},
			},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				require := require.New(t)
				require.NotNil(got.Message)
			},
		},
		"Force false should filter by latest checkpoint.": {
			mock: func() agentcontext.Compactor {
				c, _ := simple.New(simple.Config{Provider: fake.NewEchoProvider()})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("old")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("old reply")},
				{ID: "c1", Kind: model.MessageKindCompaction, Content: textContent("summary"), Compaction: &model.CompactionData{FirstKeptID: "m3"}},
				{ID: "m3", Kind: model.MessageKindUser, Content: textContent("new")},
				{ID: "m4", Kind: model.MessageKindLLM, Content: textContent("new reply")},
			},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				assert.Nil(got.Message)
				require.Len(got.Messages, 3)
				assert.Equal("c1", got.Messages[0].ID)
				assert.Equal("m3", got.Messages[1].ID)
				assert.Equal("m4", got.Messages[2].ID)
			},
		},
		"Force false with invalid checkpoint should pass through.": {
			mock: func() agentcontext.Compactor {
				c, _ := simple.New(simple.Config{Provider: fake.NewEchoProvider()})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("old")},
				{ID: "c1", Kind: model.MessageKindCompaction, Content: textContent("summary"), Compaction: &model.CompactionData{}},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("new reply")},
			},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				assert.Nil(got.Message)
				require.Len(got.Messages, 3)
				assert.Equal("m1", got.Messages[0].ID)
				assert.Equal("c1", got.Messages[1].ID)
				assert.Equal("m2", got.Messages[2].ID)
			},
		},
		"Force false with unknown checkpoint boundary should pass through.": {
			mock: func() agentcontext.Compactor {
				c, _ := simple.New(simple.Config{Provider: fake.NewEchoProvider()})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("old")},
				{ID: "c1", Kind: model.MessageKindCompaction, Content: textContent("summary"), Compaction: &model.CompactionData{FirstKeptID: "missing"}},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("new reply")},
			},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				assert.Nil(got.Message)
				require.Len(got.Messages, 3)
				assert.Equal("m1", got.Messages[0].ID)
				assert.Equal("c1", got.Messages[1].ID)
				assert.Equal("m2", got.Messages[2].ID)
			},
		},
		"Force true with empty messages should be a no-op.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return nil, fmt.Errorf("provider should not be called")
				})
				c, _ := simple.New(simple.Config{Provider: provider})
				return c
			},
			opts: agentcontext.CompactOptions{Force: true},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				assert.Nil(got.Message)
				assert.Empty(got.Messages)
			},
		},
		"Force true should create checkpoint and return compacted context.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
					_ = req
					return &llm.Response{Message: model.Message{
						Kind:     model.MessageKindLLM,
						Content:  textContent("## Goal\n- test summary"),
						Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 10, OutputTokens: 5}},
					}}, nil
				})
				c, _ := simple.New(simple.Config{Provider: provider, KeepRecentTokens: 2})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("12345678")},
				{ID: "m3", Kind: model.MessageKindUser, Content: textContent("1234")},
			},
			opts: agentcontext.CompactOptions{Force: true},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(got.Message)
				assert.Equal(model.MessageKindCompaction, got.Message.Kind)
				assert.Equal("m2", got.Message.Compaction.FirstKeptID)
				require.Len(got.Messages, 3)
				assert.Equal(model.MessageKindCompaction, got.Messages[0].Kind)
				assert.Equal("m2", got.Messages[1].ID)
				assert.Equal("m3", got.Messages[2].ID)
				assert.Equal(10, got.Usage.InputTokens)
				assert.Equal(5, got.Usage.OutputTokens)
			},
		},
		"Force true should avoid cutting at tool result.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("summary")}}, nil
				})
				c, _ := simple.New(simple.Config{Provider: provider, KeepRecentTokens: 2})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("x")},
				{ID: "m3", Kind: model.MessageKindToolResult, ToolCallID: "tc1", Content: textContent("x")},
				{ID: "m4", Kind: model.MessageKindLLM, Content: textContent("x")},
			},
			opts: agentcontext.CompactOptions{Force: true},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(got.Message)
				assert.Equal("m2", got.Message.Compaction.FirstKeptID)
				require.Len(got.Messages, 4)
				assert.Equal("m2", got.Messages[1].ID)
				assert.Equal("m3", got.Messages[2].ID)
				assert.Equal("m4", got.Messages[3].ID)
			},
		},
		"Force true should include custom instructions in summary prompt.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
					prompt := firstTextFromMessage(req.Messages[0])
					if !strings.Contains(prompt, "focus on auth") {
						return nil, fmt.Errorf("missing custom instruction in prompt")
					}
					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("summary")}}, nil
				})
				c, _ := simple.New(simple.Config{Provider: provider, KeepRecentTokens: 1})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("x")},
			},
			opts: agentcontext.CompactOptions{Force: true, CustomInstructions: "focus on auth"},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				require := require.New(t)

				require.NotNil(got.Message)
			},
		},
		"Force true should fail when summary has no text.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM}}, nil
				})
				c, _ := simple.New(simple.Config{Provider: provider, KeepRecentTokens: 1})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("x")},
			},
			opts:   agentcontext.CompactOptions{Force: true},
			expErr: true,
		},
		"Compactor should propagate summarization errors.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return nil, fmt.Errorf("boom")
				})
				c, _ := simple.New(simple.Config{Provider: provider, KeepRecentTokens: 1})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("x")},
			},
			opts:   agentcontext.CompactOptions{Force: true},
			expErr: true,
		},
		"Force true should serialize transcript prompt through public API.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
					prompt := firstTextFromMessage(req.Messages[0])

					if !strings.Contains(prompt, "[User]: user msg") {
						return nil, fmt.Errorf("missing serialized user message")
					}
					if !strings.Contains(prompt, "[LLM]: llm msg") {
						return nil, fmt.Errorf("missing serialized llm message")
					}
					if !strings.Contains(prompt, "[LLM tool calls]: read({\"path\":\"main.go\"})") {
						return nil, fmt.Errorf("missing compacted tool call arguments")
					}
					if !strings.Contains(prompt, "[Tool Result Error]: tool failed") {
						return nil, fmt.Errorf("missing serialized tool result error")
					}
					if !strings.Contains(prompt, "[Compaction Summary]: older summary") {
						return nil, fmt.Errorf("missing serialized previous compaction")
					}
					if strings.Contains(prompt, "should-not-appear") {
						return nil, fmt.Errorf("unknown kind should not be serialized")
					}

					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("summary")}}, nil
				})

				c, _ := simple.New(simple.Config{Provider: provider, KeepRecentTokens: 1})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("user msg")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("llm msg"), ToolCallRequests: []model.ToolCallRequest{{ID: "tc1", ToolID: "read", Arguments: []byte(`{ "path" : "main.go" }`)}}},
				{ID: "m3", Kind: model.MessageKindToolResult, IsError: true, ToolCallID: "tc1", Content: textContent("tool failed")},
				{ID: "c1", Kind: model.MessageKindCompaction, Content: textContent("older summary")},
				{ID: "u1", Kind: model.MessageKind("unknown"), Content: textContent("should-not-appear")},
				{ID: "m4", Kind: model.MessageKindUser, Content: textContent("latest")},
			},
			opts: agentcontext.CompactOptions{Force: true},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				require := require.New(t)
				require.NotNil(got.Message)
			},
		},
		"Force true with defaults should use default max summary tokens.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
					if req.Config.MaxTokens != defaultMaxSummaryTokens {
						return nil, fmt.Errorf("unexpected max tokens %d", req.Config.MaxTokens)
					}

					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("summary")}}, nil
				})

				c, _ := simple.New(simple.Config{Provider: provider})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent(strings.Repeat("a", 40000))},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent(strings.Repeat("b", 40000))},
				{ID: "m3", Kind: model.MessageKindUser, Content: textContent(strings.Repeat("c", 40000))},
			},
			opts: agentcontext.CompactOptions{Force: true},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				assert := assert.New(t)
				require := require.New(t)

				require.NotNil(got.Message)
				assert.Equal("m2", got.Message.Compaction.FirstKeptID)
			},
		},
		"Force false with defaults should auto-compact when threshold is exceeded.": {
			mock: func() agentcontext.Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("summary")}}, nil
				})
				c, _ := simple.New(simple.Config{Provider: provider})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent(strings.Repeat("a", 300000))},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent(strings.Repeat("b", 300000))},
				{ID: "m3", Kind: model.MessageKindUser, Content: textContent(strings.Repeat("c", 300000))},
			},
			assert: func(t *testing.T, got *agentcontext.CompactResult) {
				require := require.New(t)
				require.NotNil(got.Message)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			c := test.mock()

			ctx := test.ctx
			if ctx == nil {
				ctx = context.Background()
			}

			got, err := c.Compact(ctx, test.msgs, test.opts)
			if test.expErr {
				require.Error(err)
				return
			}

			require.NoError(err)
			require.NotNil(got)
			if test.assert != nil {
				test.assert(t, got)
			}
		})
	}
}

func textContent(text string) []model.ContentPart {
	return []model.ContentPart{{Type: model.ContentPartTypeText, Text: text}}
}

func firstTextFromMessage(msg model.Message) string {
	for _, p := range msg.Content {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			return p.Text
		}
	}

	return ""
}
