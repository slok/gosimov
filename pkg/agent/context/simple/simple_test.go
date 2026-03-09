package simple

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	agentcontext "github.com/slok/gosimov/pkg/agent/context"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/fake"
	"github.com/slok/gosimov/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	tests := map[string]struct {
		cfg    Config
		expErr bool
	}{
		"Missing provider should fail.": {
			cfg:    Config{},
			expErr: true,
		},
		"Context window lower than reserve should fail.": {
			cfg: Config{
				Provider:            fake.NewEchoProvider(),
				ContextWindowTokens: 100,
				ReserveTokens:       200,
			},
			expErr: true,
		},
		"Valid config should set defaults.": {
			cfg: Config{Provider: fake.NewEchoProvider()},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			c, err := New(test.cfg)
			if test.expErr {
				require.Error(err)
				return
			}

			require.NoError(err)
			require.NotNil(c)
			assert.Equal(defaultContextWindowTokens, c.contextWindowTokens)
			assert.Equal(defaultReserveTokens, c.reserveTokens)
			assert.Equal(defaultKeepRecentTokens, c.keepRecentTokens)
			assert.Equal(defaultMaxSummaryTokens, c.maxSummaryTokens)
		})
	}
}

func TestCompactorCompact(t *testing.T) {
	tests := map[string]struct {
		mock   func() *Compactor
		msgs   []model.Message
		opts   agentcontext.CompactOptions
		expErr bool
		assert func(t *testing.T, got *agentcontext.CompactResult)
	}{
		"Force false should pass through without checkpoint.": {
			mock: func() *Compactor {
				c, _ := New(Config{Provider: fake.NewEchoProvider()})
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
			mock: func() *Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{
						Kind:     model.MessageKindLLM,
						Content:  textContent("auto summary"),
						Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 11, OutputTokens: 7}},
					}}, nil
				})
				c, _ := New(Config{
					Provider:            provider,
					ContextWindowTokens: 6,
					ReserveTokens:       2,
					KeepRecentTokens:    2,
				})
				return c
			},
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
		"Force false should filter by latest checkpoint.": {
			mock: func() *Compactor {
				c, _ := New(Config{Provider: fake.NewEchoProvider()})
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
		"Force true should create checkpoint and return compacted context.": {
			mock: func() *Compactor {
				provider := fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{
						Kind:     model.MessageKindLLM,
						Content:  textContent("## Goal\n- test summary"),
						Metadata: &model.MessageMetadata{Usage: &model.Usage{InputTokens: 10, OutputTokens: 5}},
					}}, nil
				})
				c, _ := New(Config{Provider: provider, KeepRecentTokens: 2})
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
			mock: func() *Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("summary")}}, nil
				})
				c, _ := New(Config{Provider: provider, KeepRecentTokens: 2})
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
			mock: func() *Compactor {
				provider := fake.NewProvider(func(_ context.Context, req llm.Request) (*llm.Response, error) {
					prompt := firstText(req.Messages[0])
					if !strings.Contains(prompt, "focus on auth") {
						return nil, fmt.Errorf("missing custom instruction in prompt")
					}
					return &llm.Response{Message: model.Message{Kind: model.MessageKindLLM, Content: textContent("summary")}}, nil
				})
				c, _ := New(Config{Provider: provider, KeepRecentTokens: 1})
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
		"Compactor should propagate summarization errors.": {
			mock: func() *Compactor {
				provider := fake.NewProvider(func(_ context.Context, _ llm.Request) (*llm.Response, error) {
					return nil, fmt.Errorf("boom")
				})
				c, _ := New(Config{Provider: provider, KeepRecentTokens: 1})
				return c
			},
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser, Content: textContent("12345678")},
				{ID: "m2", Kind: model.MessageKindLLM, Content: textContent("x")},
			},
			opts:   agentcontext.CompactOptions{Force: true},
			expErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			c := test.mock()

			got, err := c.Compact(context.Background(), test.msgs, test.opts)
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

func TestSerializeMessage(t *testing.T) {
	tests := map[string]struct {
		msg model.Message
		exp string
	}{
		"User message should serialize with User tag.": {
			msg: model.Message{Kind: model.MessageKindUser, Content: textContent("hello")},
			exp: "[User]: hello",
		},
		"LLM text and tool calls should serialize both blocks.": {
			msg: model.Message{
				Kind:    model.MessageKindLLM,
				Content: textContent("planning"),
				ToolCallRequests: []model.ToolCallRequest{{
					ID:        "tc1",
					ToolID:    "read",
					Arguments: json.RawMessage(`{ "path" : "main.go" }`),
				}},
			},
			exp: "[LLM]: planning\n[LLM tool calls]: read({\"path\":\"main.go\"})",
		},
		"Tool result error should serialize with error tag.": {
			msg: model.Message{Kind: model.MessageKindToolResult, IsError: true, Content: textContent("permission denied")},
			exp: "[Tool Result Error]: permission denied",
		},
		"Compaction message should serialize with compaction tag.": {
			msg: model.Message{Kind: model.MessageKindCompaction, Content: textContent("summary")},
			exp: "[Compaction Summary]: summary",
		},
		"Unknown kind should serialize as empty.": {
			msg: model.Message{Kind: model.MessageKind("unknown")},
			exp: "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := serializeMessage(test.msg)
			assert.Equal(test.exp, got)
		})
	}
}

func TestCompactJSON(t *testing.T) {
	tests := map[string]struct {
		raw []byte
		exp string
	}{
		"Empty raw should return empty string.": {
			raw: nil,
			exp: "",
		},
		"Valid JSON should be compacted.": {
			raw: []byte(`{ "path" : "main.go", "offset": 10 }`),
			exp: `{"offset":10,"path":"main.go"}`,
		},
		"Invalid JSON should return original string.": {
			raw: []byte(`{"path":"main.go"`),
			exp: `{"path":"main.go"`,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := compactJSON(test.raw)
			assert.Equal(test.exp, got)
		})
	}
}

func TestEstimateMessageTokens(t *testing.T) {
	tests := map[string]struct {
		msg model.Message
		exp int
	}{
		"Text token estimation should use chars divided by four.": {
			msg: model.Message{Content: textContent("12345678")},
			exp: 2,
		},
		"Very short text should have minimum of one token.": {
			msg: model.Message{Content: textContent("a")},
			exp: 1,
		},
		"Tool call arguments should be included in token estimate.": {
			msg: model.Message{ToolCallRequests: []model.ToolCallRequest{{ToolID: "read", Arguments: json.RawMessage(`{"path":"main.go"}`)}}},
			exp: 5,
		},
		"Non text content should not increase token estimate.": {
			msg: model.Message{Content: []model.ContentPart{{Type: model.ContentPartTypeImage, Image: &model.ImageData{Data: []byte("abcd"), MimeType: "image/png"}}}},
			exp: 0,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := estimateMessageTokens(test.msg)
			assert.Equal(test.exp, got)
		})
	}
}

func TestApplyLatestCheckpoint(t *testing.T) {
	tests := map[string]struct {
		msgs   []model.Message
		assert func(t *testing.T, got []model.Message)
	}{
		"Missing FirstKeptID should return original messages.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{}},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
			assert: func(t *testing.T, got []model.Message) {
				assert := assert.New(t)
				require := require.New(t)

				require.Len(got, 3)
				assert.Equal("m1", got[0].ID)
				assert.Equal("c1", got[1].ID)
				assert.Equal("m2", got[2].ID)
			},
		},
		"Unknown FirstKeptID should return original messages.": {
			msgs: []model.Message{
				{ID: "m1", Kind: model.MessageKindUser},
				{ID: "c1", Kind: model.MessageKindCompaction, Compaction: &model.CompactionData{FirstKeptID: "missing"}},
				{ID: "m2", Kind: model.MessageKindLLM},
			},
			assert: func(t *testing.T, got []model.Message) {
				assert := assert.New(t)
				require := require.New(t)

				require.Len(got, 3)
				assert.Equal("m1", got[0].ID)
				assert.Equal("c1", got[1].ID)
				assert.Equal("m2", got[2].ID)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := applyLatestCheckpoint(test.msgs)
			test.assert(t, got)
		})
	}
}

func TestCompactorShouldCompact(t *testing.T) {
	tests := map[string]struct {
		compactor *Compactor
		msgs      []model.Message
		exp       bool
	}{
		"Empty messages should not compact.": {
			compactor: &Compactor{contextWindowTokens: 100, reserveTokens: 10},
			msgs:      nil,
			exp:       false,
		},
		"Below threshold should not compact.": {
			compactor: &Compactor{contextWindowTokens: 100, reserveTokens: 20},
			msgs:      []model.Message{{Content: textContent("12345678")}}, // 2 tokens
			exp:       false,
		},
		"Above threshold should compact.": {
			compactor: &Compactor{contextWindowTokens: 10, reserveTokens: 2},
			msgs: []model.Message{
				{Content: textContent("12345678")},
				{Content: textContent("12345678")},
				{Content: textContent("12345678")},
				{Content: textContent("12345678")},
				{Content: textContent("12345678")}, // 10 tokens total > 8 threshold
			},
			exp: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := test.compactor.shouldCompact(test.msgs)
			assert.Equal(test.exp, got)
		})
	}
}
