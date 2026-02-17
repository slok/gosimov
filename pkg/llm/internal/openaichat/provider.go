package openaichat

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/slok/gosimov/pkg/llm"
	llmauth "github.com/slok/gosimov/pkg/llm/internal/auth"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/pkgerrors"
	"github.com/slok/gosimov/pkg/tool"
)

type Config struct {
	TokenSource llmauth.TokenSource
	BaseURL     string
	Model       string
	ModelInfo   model.LLMModelInfo
	Tools       []tool.Tool
	ProviderID  string
	Client      *http.Client
}

func (c *Config) defaults() error {
	if c.TokenSource == nil {
		return fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}
	if c.BaseURL == "" {
		return fmt.Errorf("base url is required: %w", pkgerrors.ErrNotValid)
	}
	if c.Model == "" {
		return fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}
	if c.ProviderID == "" {
		return fmt.Errorf("provider id is required: %w", pkgerrors.ErrNotValid)
	}
	if c.Client == nil {
		c.Client = http.DefaultClient
	}
	return nil
}

type Provider struct {
	tokenSrc   llmauth.TokenSource
	baseURL    string
	model      string
	modelInfo  model.LLMModelInfo
	providerID string
	tools      []chatTool
	client     *http.Client
}

func New(cfg Config) (llm.Provider, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid openai chat provider config: %w", err)
	}

	return &Provider{
		tokenSrc:   cfg.TokenSource,
		baseURL:    cfg.BaseURL,
		model:      cfg.Model,
		modelInfo:  cfg.ModelInfo,
		providerID: cfg.ProviderID,
		tools:      convertTools(cfg.Tools),
		client:     cfg.Client,
	}, nil
}

func (p *Provider) ModelInfo() model.LLMModelInfo {
	return p.modelInfo
}

func (p *Provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	body := chatRequest{Model: p.model, Messages: convertMessages(req.SystemPrompt, req.Messages), Tools: p.tools}
	if req.Config.MaxTokens > 0 {
		body.MaxTokens = req.Config.MaxTokens
	}
	if req.Config.EnablePromptCache {
		body.PromptCacheKey = promptCacheKeyFromRequest(req)
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.baseURL, "/")+"/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("creating http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	token, err := p.tokenSrc.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting auth token: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+token)

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
		return nil, parseAPIError(httpResp.StatusCode, respBody)
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("unmarshaling response: %w", err)
	}

	msg := convertResponse(chatResp)
	if msg.Metadata == nil {
		msg.Metadata = &model.MessageMetadata{}
	}
	msg.Metadata.Provider = p.providerID

	return &llm.Response{Message: msg}, nil
}

func promptCacheKeyFromRequest(req llm.Request) string {
	sessionID := strings.TrimSpace(req.SessionID)
	if sessionID != "" {
		return "gosimov-sess-" + sessionID
	}

	payload := struct {
		SystemPrompt string          `json:"system_prompt"`
		Messages     []model.Message `json:"messages"`
	}{
		SystemPrompt: req.SystemPrompt,
		Messages:     req.Messages,
	}

	b, err := json.Marshal(payload)
	if err != nil {
		h := sha256.Sum256([]byte(req.SystemPrompt))
		return "gosimov-req-" + hex.EncodeToString(h[:8])
	}

	h := sha256.Sum256(b)
	return "gosimov-req-" + hex.EncodeToString(h[:12])
}

type chatErrorResponse struct {
	Error chatErrorDetail `json:"error"`
}

type chatErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func parseAPIError(statusCode int, body []byte) error {
	var errResp chatErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil {
		return fmt.Errorf("api error (status %d): %s: %w", statusCode, string(body), pkgerrors.ErrLLMError)
	}

	msg := strings.TrimSpace(errResp.Error.Message)
	errType := strings.TrimSpace(errResp.Error.Type)
	if msg == "" {
		msg = strings.TrimSpace(string(body))
	}
	if msg == "" {
		msg = "unknown api error"
	}

	return fmt.Errorf("api error (status %d, type=%s): %s: %w", statusCode, errType, msg, pkgerrors.ErrLLMError)
}
