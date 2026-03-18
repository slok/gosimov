// Example pr-review runs an automated pull request review using gosimov.
//
// It is designed to run in GitHub Actions and be triggered by @gosimov-review.
// The agent can read PR context and publish summary/inline comments via dedicated
// GitHub tools (no generic shell tool).
//
// Usage:
//
//	go run ./examples/pr-review --api-key <key> --repo owner/repo --pr 123
package main

import (
	"context"
	"fmt"
	"os"

	exampletools "github.com/slok/gosimov/examples/pr-review/tools"
	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/llm/opencodego"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/memory"
	"github.com/slok/gosimov/pkg/tool"
	"github.com/slok/gosimov/pkg/tool/ls"
	"github.com/slok/gosimov/pkg/tool/read"
)

const reviewMarker = "<!-- gosimov-review -->"

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	rctx, err := resolveReviewContext(cfg)
	if err != nil {
		return err
	}

	if !rctx.ShouldRun {
		fmt.Printf("Skipping: %s\n", rctx.SkipReason)
		return nil
	}

	gh := exampletools.NewGHClient(exampletools.GHClientConfig{
		Repo:    rctx.Repo,
		PR:      rctx.PRNumber,
		DryRun:  cfg.dryRun,
		Timeout: cfg.ghTimeout,
	})

	state := exampletools.NewState(exampletools.StateConfig{
		GH:                gh,
		WorkDir:           cfg.workDir,
		ReviewMarker:      reviewMarker,
		MaxInlineComments: cfg.maxInlineComments,
	})

	if err := state.Warm(ctx); err != nil {
		return fmt.Errorf("warming review state: %w", err)
	}

	toolset, err := createTools(state)
	if err != nil {
		return fmt.Errorf("creating tools: %w", err)
	}

	provider, err := opencodego.New(opencodego.Config{
		TokenSource: opencodego.NewAPIKeyTokenSource(cfg.apiKey),
		Model:       cfg.modelID,
		Tools:       toolset,
	})
	if err != nil {
		return fmt.Errorf("creating provider: %w", err)
	}

	repo := memory.NewRepository()

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		SystemPrompt:      buildSystemPrompt(reviewMarker, cfg.maxInlineComments),
		Tools:             toolset,
		SessionRepository: repo,
		MessageRepository: repo,
		TurnMaxIterations: cfg.maxIterations,
	})
	if err != nil {
		return fmt.Errorf("creating session: %w", err)
	}

	userPrompt := buildUserPrompt(rctx, cfg)
	fmt.Printf("Repo:      %s\n", rctx.Repo)
	fmt.Printf("PR:        #%d\n", rctx.PRNumber)
	fmt.Printf("Model:     %s\n", cfg.modelID)
	fmt.Printf("Dry run:   %t\n", cfg.dryRun)
	fmt.Printf("Work dir:  %s\n", cfg.workDir)
	fmt.Printf("Session:   %s\n\n", session.Session().ID)

	result, err := session.Prompt(ctx, []model.ContentPart{model.NewContentText(userPrompt)}, agent.PromptOptions{})
	if err != nil {
		return fmt.Errorf("review prompt failed: %w", err)
	}

	if len(result.Message.Content) > 0 {
		fmt.Println("LLM final message:")
		fmt.Println(result.Message.Content[0].Text)
	}

	fmt.Printf("\nInline comments posted: %d\n", state.InlineCommentsPosted())
	if state.SummaryCommentAction() != "" {
		fmt.Printf("Summary comment: %s\n", state.SummaryCommentAction())
	}

	return nil
}

func createTools(state *exampletools.State) ([]tool.Tool, error) {
	lsTool, err := ls.New(ls.Config{CWD: state.WorkDir()})
	if err != nil {
		return nil, fmt.Errorf("creating ls tool: %w", err)
	}

	readTool, err := read.New(read.Config{CWD: state.WorkDir()})
	if err != nil {
		return nil, fmt.Errorf("creating read tool: %w", err)
	}

	tools := []tool.Tool{
		exampletools.WrapWithLogging(lsTool),
		exampletools.WrapWithLogging(readTool),
		exampletools.WrapWithLogging(exampletools.NewPROverviewReadTool(state)),
		exampletools.WrapWithLogging(exampletools.NewPRDiffReadTool(state)),
		exampletools.WrapWithLogging(exampletools.NewPRFileReadTool(state)),
		exampletools.WrapWithLogging(exampletools.NewPRDiscussionReadTool(state)),
		exampletools.WrapWithLogging(exampletools.NewPRCommentUpsertTool(state)),
		exampletools.WrapWithLogging(exampletools.NewPRInlineCommentCreateTool(state)),
	}

	return tools, nil
}
