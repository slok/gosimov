package openai

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
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
	defaultChatGPTBaseURL = "https://chatgpt.com/backend-api"
	chatGPTProviderID     = "openai-codex"
	chatGPTOriginator     = "pi"
)

// ChatGPTConfig configures the ChatGPT Codex responses provider.
type ChatGPTConfig struct {
	// TokenSource provides bearer tokens for authentication (required).
	// Use [NewAPIKeyTokenSource] for static API key auth.
	TokenSource TokenSource
	// BaseURL overrides ChatGPT API base URL.
	// Defaults to "https://chatgpt.com/backend-api".
	BaseURL string
	// Model is the model ID to use (e.g. "gpt-5.3-codex") (required).
	Model string
	// Client is the HTTP client used for API calls (optional).
	// Defaults to [http.DefaultClient].
	Client *http.Client
}

func (c *ChatGPTConfig) defaults() error {
	if c.TokenSource == nil {
		return fmt.Errorf("token source is required: %w", pkgerrors.ErrNotValid)
	}

	if c.Model == "" {
		return fmt.Errorf("model is required: %w", pkgerrors.ErrNotValid)
	}

	if c.BaseURL == "" {
		c.BaseURL = defaultChatGPTBaseURL
	}

	if c.Client == nil {
		c.Client = http.DefaultClient
	}

	if !IsSupportedChatGPTModel(c.Model) {
		return fmt.Errorf("unsupported chatgpt model %q: %w", c.Model, pkgerrors.ErrNotValid)
	}

	return nil
}

type chatGPTProvider struct {
	tokenSrc   TokenSource
	baseURL    string
	model      string
	modelInfo  model.LLMModelInfo
	providerID string
	originator string
	client     *http.Client
}

// NewChatGPT creates a new ChatGPT Codex responses provider.
func NewChatGPT(cfg ChatGPTConfig) (llm.Provider, error) {
	if err := cfg.defaults(); err != nil {
		return nil, fmt.Errorf("invalid codex responses provider config: %w", err)
	}

	info, _ := ChatGPTModelInfo(cfg.Model)

	return &chatGPTProvider{
		tokenSrc:   cfg.TokenSource,
		baseURL:    cfg.BaseURL,
		model:      cfg.Model,
		modelInfo:  info,
		providerID: chatGPTProviderID,
		originator: chatGPTOriginator,
		client:     cfg.Client,
	}, nil
}

func (p *chatGPTProvider) ModelInfo() model.LLMModelInfo {
	return p.modelInfo
}

// Call sends a responses request and returns the full response.
func (p *chatGPTProvider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	body := codexRequest{
		Model:             p.model,
		Store:             false,
		Stream:            true,
		Instructions:      req.SystemPrompt,
		Input:             convertCodexMessages(req.Messages),
		Tools:             convertCodexTools(req.Tools),
		Text:              &codexText{Verbosity: "medium"},
		Include:           []string{"reasoning.encrypted_content"},
		ToolChoice:        "auto",
		ParallelToolCalls: true,
	}

	if req.Config.EnablePromptCache {
		// Keep cache key provider-agnostic for callers by deriving a deterministic
		// key from request content instead of exposing provider-specific key fields.
		body.PromptCacheKey = promptCacheKeyFromRequest(req)
		if strings.Contains(strings.ToLower(p.baseURL), "api.openai.com") {
			body.PromptCacheTTL = "24h"
		}
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, resolveCodexEndpoint(p.baseURL), bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("creating http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("OpenAI-Beta", "responses=experimental")
	httpReq.Header.Set("originator", p.originator)

	token, err := p.tokenSrc.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting auth token: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+token)

	accountID, err := extractChatGPTAccountID(token)
	if err != nil {
		return nil, fmt.Errorf("extracting chatgpt account id from token: %w", err)
	}
	httpReq.Header.Set("chatgpt-account-id", accountID)

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

	codexResp, err := parseCodexSuccessResponse(respBody)
	if err != nil {
		return nil, fmt.Errorf("parsing codex response: %w", err)
	}

	msg := convertCodexResponse(codexResp)
	if msg.Metadata == nil {
		msg.Metadata = &model.MessageMetadata{}
	}
	msg.Metadata.Provider = p.providerID

	return &llm.Response{Message: msg}, nil
}

func promptCacheKeyFromRequest(req llm.Request) string {
	toolFingerprint := toolDefinitionsFingerprint(req.Tools)

	sessionID := strings.TrimSpace(req.SessionID)
	if sessionID != "" {
		if toolFingerprint == "" {
			return "gosimov-sess-" + sessionID
		}

		return "gosimov-sess-" + sessionID + "-tools-" + toolFingerprint
	}

	payload := struct {
		SystemPrompt string               `json:"system_prompt"`
		Messages     []model.Message      `json:"messages"`
		Tools        []llm.ToolDefinition `json:"tools,omitempty"`
	}{
		SystemPrompt: req.SystemPrompt,
		Messages:     req.Messages,
		Tools:        req.Tools,
	}

	b, err := json.Marshal(payload)
	if err != nil {
		h := sha256.Sum256([]byte(req.SystemPrompt))
		return "gosimov-req-" + hex.EncodeToString(h[:8])
	}

	h := sha256.Sum256(b)
	return "gosimov-req-" + hex.EncodeToString(h[:12])
}

func toolDefinitionsFingerprint(tools []llm.ToolDefinition) string {
	if len(tools) == 0 {
		return ""
	}

	b, err := json.Marshal(tools)
	if err != nil {
		h := sha256.Sum256([]byte(fmt.Sprintf("tool-count-%d", len(tools))))
		return hex.EncodeToString(h[:6])
	}

	h := sha256.Sum256(b)

	return hex.EncodeToString(h[:6])
}

func parseCodexSuccessResponse(body []byte) (codexResponse, error) {
	// Some compatible backends may still return plain JSON.
	var direct codexResponse
	if err := json.Unmarshal(body, &direct); err == nil && direct.Model != "" {
		return direct, nil
	}

	chunks := strings.Split(string(body), "\n\n")
	var lastResp *codexResponse
	for _, chunk := range chunks {
		chunk = strings.TrimSpace(chunk)
		if chunk == "" {
			continue
		}

		lines := strings.Split(chunk, "\n")
		dataLines := make([]string, 0, len(lines))
		for _, ln := range lines {
			ln = strings.TrimSpace(ln)
			if strings.HasPrefix(ln, "data:") {
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(ln, "data:")))
			}
		}

		if len(dataLines) == 0 {
			continue
		}

		data := strings.Join(dataLines, "\n")
		if data == "" || data == "[DONE]" {
			continue
		}

		var evt struct {
			Type     string          `json:"type"`
			Message  string          `json:"message,omitempty"`
			Code     string          `json:"code,omitempty"`
			Response json.RawMessage `json:"response,omitempty"`
		}
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			continue
		}

		if evt.Type == "error" {
			msg := strings.TrimSpace(evt.Message)
			if msg == "" {
				msg = strings.TrimSpace(data)
			}
			return codexResponse{}, fmt.Errorf("stream error (%s): %s: %w", evt.Code, msg, pkgerrors.ErrLLMError)
		}

		if evt.Type == "response.completed" || evt.Type == "response.done" {
			if len(evt.Response) == 0 {
				continue
			}
			var r codexResponse
			if err := json.Unmarshal(evt.Response, &r); err != nil {
				continue
			}
			lastResp = &r
		}
	}

	if lastResp == nil {
		return codexResponse{}, fmt.Errorf("missing response.completed event: %w", pkgerrors.ErrLLMError)
	}

	return *lastResp, nil
}

func resolveCodexEndpoint(baseURL string) string {
	normalized := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(normalized, "/codex/responses") {
		return normalized
	}
	if strings.HasSuffix(normalized, "/codex") {
		return normalized + "/responses"
	}

	return normalized + "/codex/responses"
}

func extractChatGPTAccountID(token string) (string, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("token is not a jwt: %w", pkgerrors.ErrNotValid)
	}

	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("decoding jwt payload: %w", pkgerrors.ErrNotValid)
	}

	var claims map[string]any
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("unmarshaling jwt payload: %w", pkgerrors.ErrNotValid)
	}

	authData, ok := claims["https://api.openai.com/auth"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("missing auth claim: %w", pkgerrors.ErrNotValid)
	}

	accountID, _ := authData["chatgpt_account_id"].(string)
	if strings.TrimSpace(accountID) == "" {
		return "", fmt.Errorf("missing chatgpt account id claim: %w", pkgerrors.ErrNotValid)
	}

	return accountID, nil
}
