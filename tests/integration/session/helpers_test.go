package session_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/llm"
	"github.com/slok/gosimov/pkg/llm/opencodego"
	"github.com/slok/gosimov/pkg/model"
)

// Config holds integration test configuration loaded from environment variables.
//
// Environment variables:
//
//	GOSIMOV_INTEGRATION           - must be "true" to enable integration tests
//	INTEGRATION_OPENCODE_GO_API_KEY         - required, OpenCode Go API key
//	INTEGRATION_OPENCODE_GO_MODEL           - optional, model for conversation
//	OPENCODEGO_INTEGRATION_SUMMARY_MODEL    - optional, model for compaction summarization
//	OPENCODEGO_INTEGRATION_TIMEOUT          - optional, per-prompt timeout as Go duration
type Config struct {
	APIKey       string
	Model        string
	SummaryModel string
	Timeout      time.Duration
}

func (c *Config) defaults() error {
	if c.APIKey == "" {
		return fmt.Errorf("INTEGRATION_OPENCODE_GO_API_KEY is required")
	}

	if c.Model == "" {
		c.Model = opencodego.ModelMinimaxM25
	}

	if !opencodego.IsSupportedModel(c.Model) {
		return fmt.Errorf("unsupported model %q", c.Model)
	}

	if c.SummaryModel == "" {
		c.SummaryModel = c.Model
	}

	if !opencodego.IsSupportedModel(c.SummaryModel) {
		return fmt.Errorf("unsupported summary model %q", c.SummaryModel)
	}

	if c.Timeout == 0 {
		c.Timeout = 3 * time.Minute
	}

	return nil
}

// NewConfig loads integration test configuration from environment variables.
// If the activation env var is not set or the config is invalid, the test is skipped.
func NewConfig(t *testing.T) Config {
	t.Helper()

	const (
		envActivation   = "GOSIMOV_INTEGRATION"
		envAPIKey       = "INTEGRATION_OPENCODE_GO_API_KEY"
		envModel        = "INTEGRATION_OPENCODE_GO_MODEL"
		envSummaryModel = "OPENCODEGO_INTEGRATION_SUMMARY_MODEL"
		envTimeout      = "OPENCODEGO_INTEGRATION_TIMEOUT"
	)

	if os.Getenv(envActivation) != "true" {
		t.Skipf("skipping integration test: %s is not set to \"true\"", envActivation)
	}

	c := Config{
		APIKey:       os.Getenv(envAPIKey),
		Model:        os.Getenv(envModel),
		SummaryModel: os.Getenv(envSummaryModel),
	}

	if v := os.Getenv(envTimeout); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			t.Skipf("skipping: invalid %s=%q: %v", envTimeout, v, err)
		}
		c.Timeout = d
	}

	if err := c.defaults(); err != nil {
		t.Skipf("skipping: invalid config: %s", err)
	}

	return c
}

// NewProvider creates an OpenCode Go LLM provider using the config's API key and model.
func (c Config) NewProvider(t *testing.T) llm.Provider {
	t.Helper()

	p, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource(c.APIKey),
		Model:       c.Model,
	})
	if err != nil {
		t.Fatalf("creating opencode-go provider: %v", err)
	}

	return p
}

// NewSummaryProvider creates an OpenCode Go LLM provider for compaction summarization.
func (c Config) NewSummaryProvider(t *testing.T) llm.Provider {
	t.Helper()

	p, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource(c.APIKey),
		Model:       c.SummaryModel,
	})
	if err != nil {
		t.Fatalf("creating opencode-go summary provider: %v", err)
	}

	return p
}

func promptWithRetry(t *testing.T, ctx context.Context, session *agent.Session, content []model.ContentPart) (*agent.SessionTurnResult, error) {
	t.Helper()

	const maxAttempts = 3

	var err error
	for i := 0; i < maxAttempts; i++ {
		res, promptErr := session.Prompt(ctx, content, agent.PromptOptions{})
		if promptErr == nil {
			return res, nil
		}

		err = promptErr
		if !isRetryableIntegrationErr(promptErr) || i == maxAttempts-1 {
			return nil, promptErr
		}

		time.Sleep(2 * time.Second)
	}

	return nil, err
}

func isRetryableIntegrationErr(err error) bool {
	if err == nil {
		return false
	}

	errMsg := err.Error()
	return strings.Contains(errMsg, "status 429") || strings.Contains(errMsg, "status 500") || strings.Contains(errMsg, "rate limit")
}

func finalLLMMessageFromTurnResult(t *testing.T, result *agent.SessionTurnResult) model.Message {
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
