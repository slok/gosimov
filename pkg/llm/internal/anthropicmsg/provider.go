package anthropicmsg

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

const defaultVersionHeader = "2023-06-01"

type AuthMode int

const (
	AuthModeAPIKey AuthMode = iota
	AuthModeOAuthBearer
)

type Options struct {
	ProviderID         string
	AuthMode           AuthMode
	ClaudeCompat       bool
	NormalizeToolName  func(string) string
	ExtraHeaders       map[string]string
	DefaultMaxTokens   int
	ClaudeIdentityText string
}

type Config struct {
	TokenSource llmauth.TokenSource
	BaseURL     string
	Model       string
	ModelInfo   model.LLMModelInfo
	Client      *http.Client
	Options     Options
}

func (c *Config) defaults() error {
	if c.TokenSource == nil {
		return fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.BaseURL) == "" {
		return fmt.Errorf("base url is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(c.Options.ProviderID) == "" {
		return fmt.Errorf("provider id is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	if c.Options.DefaultMaxTokens <= 0 {
		c.Options.DefaultMaxTokens = 4096
	}

	return nil
}

type Provider struct {
	tokenSrc  llmauth.TokenSource
	baseURL   string
	model     string
	modelInfo model.LLMModelInfo
	client    *http.Client

	opts Options
}

func New(cfg Config) (llm.Provider, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid anthropic messages provider config: %w", err)
	}

	return &Provider{
		tokenSrc:  cfg.TokenSource,
		baseURL:   strings.TrimRight(cfg.BaseURL, "/"),
		model:     cfg.Model,
		modelInfo: cfg.ModelInfo,
		client:    cfg.Client,
		opts:      cfg.Options,
	}, nil
}

func (p *Provider) ModelInfo() model.LLMModelInfo {
	return p.modelInfo
}

func (p *Provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	body := anthropicRequest{
		Model:     p.model,
		Messages:  convertMessages(req.Messages, p.opts.NormalizeToolName),
		Tools:     convertTools(req.Tools, p.opts.NormalizeToolName),
		MaxTokens: p.opts.DefaultMaxTokens,
	}

	if req.Config.MaxTokens > 0 {
		body.MaxTokens = req.Config.MaxTokens
	}

	systemPrompt := ""
	if strings.TrimSpace(req.SystemPrompt) != "" {
		if p.opts.ClaudeCompat && strings.TrimSpace(p.opts.ClaudeIdentityText) != "" {
			systemPrompt = p.opts.ClaudeIdentityText + "\n\n" + req.SystemPrompt
		} else {
			systemPrompt = req.SystemPrompt
		}
	} else if p.opts.ClaudeCompat && strings.TrimSpace(p.opts.ClaudeIdentityText) != "" {
		systemPrompt = p.opts.ClaudeIdentityText
	}

	if strings.TrimSpace(systemPrompt) != "" {
		body.System = systemPrompt
	}

	if req.Config.EnablePromptCache {
		applyPromptCacheControl(&body, p.baseURL)
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("creating http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", defaultVersionHeader)

	for k, v := range p.opts.ExtraHeaders {
		httpReq.Header.Set(k, v)
	}

	token, err := p.tokenSrc.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting auth token: %w", err)
	}

	switch p.opts.AuthMode {
	case AuthModeAPIKey:
		httpReq.Header.Set("x-api-key", token)
	case AuthModeOAuthBearer:
		httpReq.Header.Set("Authorization", "Bearer "+token)
	default:
		return nil, fmt.Errorf("unsupported auth mode: %w", pkgerrors.ErrNotValid)
	}

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: %w", parseAPIError(httpResp.StatusCode, respBody), pkgerrors.ErrLLMError)
	}

	var apiResp anthropicResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return nil, fmt.Errorf("unmarshaling response: %w", err)
	}

	msg := convertResponse(apiResp, restoreToolNameFromDefinitions(req.Tools, p.opts.NormalizeToolName))
	if msg.Metadata == nil {
		msg.Metadata = &model.MessageMetadata{}
	}
	msg.Metadata.Provider = p.opts.ProviderID

	return &llm.Response{Message: msg}, nil
}

func restoreToolNameFromDefinitions(tools []llm.ToolDefinition, normalizeName func(string) string) func(string) string {
	if len(tools) == 0 {
		return nil
	}

	byNormalizedName := make(map[string]string, len(tools))
	for _, t := range tools {
		name := strings.TrimSpace(t.ID)
		if name == "" {
			continue
		}

		normalized := name
		if normalizeName != nil {
			normalized = normalizeName(name)
		}

		byNormalizedName[strings.ToLower(normalized)] = name
	}

	if len(byNormalizedName) == 0 {
		return nil
	}

	return func(name string) string {
		if originalName, ok := byNormalizedName[strings.ToLower(name)]; ok {
			return originalName
		}

		return name
	}
}

func applyPromptCacheControl(body *anthropicRequest, baseURL string) {
	if body == nil {
		return
	}

	cc := &anthropicCacheControl{Type: "ephemeral"}
	if strings.Contains(baseURL, "api.anthropic.com") {
		cc.TTL = "1h"
	}

	switch system := body.System.(type) {
	case string:
		if strings.TrimSpace(system) != "" {
			body.System = []anthropicTextBlock{{Type: "text", Text: system, CacheControl: cc}}
		}
	case []anthropicTextBlock:
		if len(system) > 0 {
			system[0].CacheControl = cc
			body.System = system
		}
	}

	if len(body.Messages) == 0 {
		return
	}

	last := body.Messages[len(body.Messages)-1]
	if last.Role != "user" {
		return
	}

	switch content := last.Content.(type) {
	case string:
		if strings.TrimSpace(content) == "" {
			return
		}
		last.Content = []anthropicTextBlock{{Type: "text", Text: content, CacheControl: cc}}
	case []anthropicToolResultBlock:
		if len(content) == 0 {
			return
		}
		content[len(content)-1].CacheControl = cc
		last.Content = content
	case []any:
		for i := len(content) - 1; i >= 0; i-- {
			switch b := content[i].(type) {
			case anthropicTextBlock:
				b.CacheControl = cc
				content[i] = b
				last.Content = content
				body.Messages[len(body.Messages)-1] = last
				return
			case anthropicToolResultBlock:
				b.CacheControl = cc
				content[i] = b
				last.Content = content
				body.Messages[len(body.Messages)-1] = last
				return
			}
		}
		return
	default:
		return
	}

	body.Messages[len(body.Messages)-1] = last
}
