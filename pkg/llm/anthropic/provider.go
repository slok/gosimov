package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
)

const (
	defaultAnthropicBaseURL = "https://api.anthropic.com/v1"
	defaultVersionHeader    = "2023-06-01"
)

type authMode int

const (
	authModeAPIKey authMode = iota
	authModeOAuthBearer
)

type providerOptions struct {
	providerID         string
	authMode           authMode
	claudeCompat       bool
	normalizeToolName  func(string) string
	restoreToolName    func(string) string
	extraHeaders       map[string]string
	defaultMaxTokens   int
	claudeIdentityText string
}

type provider struct {
	tokenSrc  TokenSource
	baseURL   string
	model     string
	modelInfo model.LLMModelInfo
	tools     []anthropicTool
	client    *http.Client

	opts providerOptions
}

type providerConfig struct {
	TokenSource TokenSource
	BaseURL     string
	Model       string
	ModelInfo   model.LLMModelInfo
	Tools       []anthropicTool
	Client      *http.Client

	Options providerOptions
}

func newProvider(cfg providerConfig) (llm.Provider, error) {
	if cfg.TokenSource == nil {
		return nil, fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, fmt.Errorf("base url is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(cfg.Model) == "" {
		return nil, fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if strings.TrimSpace(cfg.Options.providerID) == "" {
		return nil, fmt.Errorf("provider id is required: %w", pkgerrors.ErrNotValid)
	}

	if cfg.Client == nil {
		cfg.Client = http.DefaultClient
	}

	if cfg.Options.defaultMaxTokens <= 0 {
		cfg.Options.defaultMaxTokens = 4096
	}

	return &provider{
		tokenSrc:  cfg.TokenSource,
		baseURL:   strings.TrimRight(cfg.BaseURL, "/"),
		model:     cfg.Model,
		modelInfo: cfg.ModelInfo,
		tools:     cfg.Tools,
		client:    cfg.Client,
		opts:      cfg.Options,
	}, nil
}

func (p *provider) ModelInfo() model.LLMModelInfo {
	return p.modelInfo
}

func (p *provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	body := anthropicRequest{
		Model:     p.model,
		Messages:  convertMessages(req.Messages, p.opts.normalizeToolName),
		Tools:     p.tools,
		MaxTokens: p.opts.defaultMaxTokens,
	}

	if req.Config.MaxTokens > 0 {
		body.MaxTokens = req.Config.MaxTokens
	}

	systemPrompt := ""
	if strings.TrimSpace(req.SystemPrompt) != "" {
		if p.opts.claudeCompat && strings.TrimSpace(p.opts.claudeIdentityText) != "" {
			systemPrompt = p.opts.claudeIdentityText + "\n\n" + req.SystemPrompt
		} else {
			systemPrompt = req.SystemPrompt
		}
	} else if p.opts.claudeCompat && strings.TrimSpace(p.opts.claudeIdentityText) != "" {
		systemPrompt = p.opts.claudeIdentityText
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

	for k, v := range p.opts.extraHeaders {
		httpReq.Header.Set(k, v)
	}

	token, err := p.tokenSrc.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting auth token: %w", err)
	}

	switch p.opts.authMode {
	case authModeAPIKey:
		httpReq.Header.Set("x-api-key", token)
	case authModeOAuthBearer:
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

	msg := convertResponse(apiResp, p.opts.restoreToolName)
	if msg.Metadata == nil {
		msg.Metadata = &model.MessageMetadata{}
	}
	msg.Metadata.Provider = p.opts.providerID

	return &llm.Response{Message: msg}, nil
}

func applyPromptCacheControl(body *anthropicRequest, baseURL string) {
	if body == nil {
		return
	}

	cc := &anthropicCacheControl{Type: "ephemeral"}
	// Anthropic supports longer-lived cache retention through ttl on direct API calls.
	// Other Anthropic-compatible gateways may reject ttl values, so keep ephemeral only there.
	if strings.Contains(baseURL, "api.anthropic.com") {
		cc.TTL = "1h"
	}

	switch system := body.System.(type) {
	case string:
		// System can be encoded as a plain string by default; convert it to
		// a text block so Anthropic cache_control can be attached.
		if strings.TrimSpace(system) != "" {
			body.System = []anthropicTextBlock{{Type: "text", Text: system, CacheControl: cc}}
		}
	case []anthropicTextBlock:
		// System can also be encoded as Anthropic text blocks; annotate the
		// first block to keep payload changes minimal while enabling caching.
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
		// Single-part user input is often encoded as a string; convert to a
		// text block because cache_control is a block-level field.
		if strings.TrimSpace(content) == "" {
			return
		}
		last.Content = []anthropicTextBlock{{Type: "text", Text: content, CacheControl: cc}}
	case []anthropicToolResultBlock:
		// Consecutive tool results are encoded as tool_result blocks. Annotate
		// the latest block to bias caching toward the newest stable boundary.
		if len(content) == 0 {
			return
		}
		content[len(content)-1].CacheControl = cc
		last.Content = content
	case []any:
		// Mixed multipart content (text/tool results/images). Walk backward and
		// annotate the last cacheable block to preserve prior structure/order.
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
