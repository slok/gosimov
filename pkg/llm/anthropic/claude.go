package anthropic

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

const (
	claudeProviderID            = "anthropic-claude"
	claudeCLIUserAgent          = "claude-cli/2.1.62"
	claudeIdentitySystemPrompt  = "You are Claude Code, Anthropic's official CLI for Claude."
	claudeCompatibilityBetaList = "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"
)

// ClaudeConfig configures the Claude Pro/Max provider using OAuth authentication.
type ClaudeConfig struct {
	TokenSource TokenSource
	BaseURL     string
	Model       string
	Tools       []tool.Tool
	Client      *http.Client
}

func (c *ClaudeConfig) defaults() (model.LLMModelInfo, error) {
	if c.TokenSource == nil {
		return model.LLMModelInfo{}, fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.Model) == "" {
		return model.LLMModelInfo{}, fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	info, ok := ModelByID(c.Model)
	if !ok {
		return model.LLMModelInfo{}, fmt.Errorf("unsupported anthropic model %q: %w", c.Model, pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.BaseURL) == "" {
		c.BaseURL = defaultAnthropicBaseURL
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	return info, nil
}

// NewClaude creates a Claude Pro/Max compatible provider using OAuth authentication.
func NewClaude(cfg ClaudeConfig) (llm.Provider, error) {
	info, err := cfg.defaults()
	if err != nil {
		return nil, fmt.Errorf("invalid claude provider config: %w", err)
	}

	return newProvider(providerConfig{
		TokenSource: cfg.TokenSource,
		BaseURL:     cfg.BaseURL,
		Model:       cfg.Model,
		ModelInfo:   info,
		Tools:       convertTools(cfg.Tools, toClaudeCodeName),
		Client:      cfg.Client,
		Options: providerOptions{
			providerID:         claudeProviderID,
			authMode:           authModeOAuthBearer,
			claudeCompat:       true,
			normalizeToolName:  toClaudeCodeName,
			restoreToolName:    fromClaudeCodeName(cfg.Tools),
			defaultMaxTokens:   info.MaxOutputTokens,
			claudeIdentityText: claudeIdentitySystemPrompt,
			extraHeaders: map[string]string{
				"anthropic-beta": claudeCompatibilityBetaList,
				"user-agent":     claudeCLIUserAgent,
				"x-app":          "cli",
			},
		},
	})
}

var claudeCodeTools = []string{
	"Read",
	"Write",
	"Edit",
	"Bash",
	"Grep",
	"Glob",
	"AskUserQuestion",
	"EnterPlanMode",
	"ExitPlanMode",
	"KillShell",
	"NotebookEdit",
	"Skill",
	"Task",
	"TaskOutput",
	"TodoWrite",
	"WebFetch",
	"WebSearch",
}

var claudeCodeToolMap = func() map[string]string {
	out := make(map[string]string, len(claudeCodeTools))
	for _, name := range claudeCodeTools {
		out[strings.ToLower(name)] = name
	}

	return out
}()

func toClaudeCodeName(name string) string {
	if v, ok := claudeCodeToolMap[strings.ToLower(name)]; ok {
		return v
	}

	return name
}

func fromClaudeCodeName(tools []tool.Tool) func(string) string {
	byLowerName := make(map[string]string, len(tools))
	for _, t := range tools {
		byLowerName[strings.ToLower(t.ID())] = t.ID()
	}

	return func(name string) string {
		if v, ok := byLowerName[strings.ToLower(name)]; ok {
			return v
		}

		return name
	}
}
