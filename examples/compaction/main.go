// Example compaction demonstrates a multi-turn session with forced compaction
// using the real simple compactor implementation and a real Zen/OpenAI provider.
//
// Usage:
//
//	ZEN_API_KEY=<key> go run ./examples/compaction
//	ZEN_API_KEY=<key> ZEN_MODEL=kimi-k2.5-free go run ./examples/compaction
//	ZEN_API_KEY=<key> ZEN_MODEL=kimi-k2.5-free ZEN_SUMMARY_MODEL=glm-5-free go run ./examples/compaction
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/slok/gosimov/pkg/agent"
	"github.com/slok/gosimov/pkg/agent/context/simple"
	"github.com/slok/gosimov/pkg/llm/zen"
	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/store/jsonl"
)

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	apiKey := os.Getenv("ZEN_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("ZEN_API_KEY environment variable is required")
	}

	modelID := os.Getenv("ZEN_MODEL")
	if modelID == "" {
		modelID = "kimi-k2.5-free"
	}

	summaryModelID := os.Getenv("ZEN_SUMMARY_MODEL")
	if summaryModelID == "" {
		summaryModelID = modelID
	}

	stepTimeout, err := timeoutFromEnv("ZEN_STEP_TIMEOUT", 5*time.Minute)
	if err != nil {
		return err
	}

	httpClient := &http.Client{Timeout: stepTimeout}

	provider, err := zen.New(zen.Config{
		TokenSource: zen.NewAPIKeyTokenSource(apiKey),
		Model:       modelID,
		Client:      httpClient,
	})
	if err != nil {
		return fmt.Errorf("creating conversation provider: %w", err)
	}

	summaryProvider, err := zen.New(zen.Config{
		TokenSource: zen.NewAPIKeyTokenSource(apiKey),
		Model:       summaryModelID,
		Client:      httpClient,
	})
	if err != nil {
		return fmt.Errorf("creating summary provider: %w", err)
	}

	compactor, err := simple.New(simple.Config{
		Provider:         summaryProvider,
		KeepRecentTokens: 120,
	})
	if err != nil {
		return fmt.Errorf("creating compactor: %w", err)
	}

	// Create the repository
	repo, err := jsonl.New(jsonl.Config{Dir: "/tmp"})
	if err != nil {
		return fmt.Errorf("creating repository: %w", err)
	}

	session, err := agent.NewSession(ctx, agent.SessionConfig{
		Provider:          provider,
		Compactor:         compactor,
		SystemPrompt:      "You are a pragmatic software engineer. Be concise. Never request tool calls.",
		SessionRepository: repo,
		MessageRepository: repo,
		TurnMaxIterations: 8,
	})
	if err != nil {
		return fmt.Errorf("creating session: %w", err)
	}

	fmt.Printf("Model:          %s\n", modelID)
	fmt.Printf("Summary model:  %s\n", summaryModelID)
	fmt.Printf("Step timeout:   %s\n", stepTimeout)
	fmt.Printf("Session:        %s\n\n", session.Session().ID)

	// Use per-turn timeouts so the example never hangs indefinitely.
	timeout := stepTimeout

	turn1Prompt := "I am moving REST handlers to a service layer in Go. Give me a migration plan that keeps behavior parity, minimizes risk, and stays incremental."
	fmt.Println("Calling turn 1...")
	turn1Start := time.Now()
	turn1Ctx, cancelTurn1 := context.WithTimeout(ctx, timeout)
	turn1, err := session.Prompt(turn1Ctx, textParts(turn1Prompt), agent.PromptOptions{})
	cancelTurn1()
	if err != nil {
		return fmt.Errorf("turn 1: %w", err)
	}
	fmt.Printf("Turn 1 done in %s\n\n", time.Since(turn1Start).Round(time.Millisecond))
	printTurn("Turn 1", turn1)

	turn2Prompt := "Now propose concrete implementation steps, call out risks around validation and error mapping, and explain what to do first for safe rollout."
	fmt.Println("Calling turn 2...")
	turn2Start := time.Now()
	turn2Ctx, cancelTurn2 := context.WithTimeout(ctx, timeout)
	turn2, err := session.Prompt(turn2Ctx, textParts(turn2Prompt), agent.PromptOptions{})
	cancelTurn2()
	if err != nil {
		return fmt.Errorf("turn 2: %w", err)
	}
	fmt.Printf("Turn 2 done in %s\n\n", time.Since(turn2Start).Round(time.Millisecond))
	printTurn("Turn 2", turn2)

	fmt.Println("--- Forced Compaction ---")
	fmt.Println("Calling compaction...")
	compactStart := time.Now()
	compactCtx, cancelCompact := context.WithTimeout(ctx, timeout)
	compactResult, err := session.Compact(compactCtx)
	cancelCompact()
	if err != nil {
		return fmt.Errorf("forcing compaction: %w", err)
	}
	fmt.Printf("Compaction done in %s\n", time.Since(compactStart).Round(time.Millisecond))
	if compactResult.Message == nil {
		fmt.Println("No checkpoint generated (context too small for current keepRecentTokens).")
	} else {
		fmt.Printf("Checkpoint ID:  %s\n", compactResult.Message.ID)
		fmt.Printf("First kept ID:  %s\n", compactResult.Message.Compaction.FirstKeptID)
		fmt.Printf("Tokens before:  %d\n", compactResult.Message.Compaction.TokensBefore)
		fmt.Printf("Summary:        %s\n", truncate(firstText(*compactResult.Message), 110))
	}
	fmt.Println()

	turn3Prompt := "Great, what tests should I write first after this refactor? Please prioritize by risk and include at least one integration and one unit test example."
	fmt.Println("Calling turn 3...")
	turn3Start := time.Now()
	turn3Ctx, cancelTurn3 := context.WithTimeout(ctx, timeout)
	turn3, err := session.Prompt(turn3Ctx, textParts(turn3Prompt), agent.PromptOptions{})
	cancelTurn3()
	if err != nil {
		return fmt.Errorf("turn 3: %w", err)
	}
	fmt.Printf("Turn 3 done in %s\n\n", time.Since(turn3Start).Round(time.Millisecond))
	printTurn("Turn 3", turn3)

	fmt.Println("--- Session Messages ---")
	for i, msg := range session.Messages() {
		fmt.Printf("%02d. %s", i+1, msg.Kind)
		if msg.Kind == model.MessageKindCompaction && msg.Compaction != nil {
			fmt.Printf(" (first_kept=%s)", msg.Compaction.FirstKeptID)
		}
		if text := firstText(msg); text != "" {
			fmt.Printf(": %s", truncate(text, 95))
		}
		fmt.Println()
	}

	u := session.Usage()
	fmt.Println("\n--- Usage ---")
	fmt.Printf("Total tokens:  %d\n", u.TotalTokens)
	fmt.Printf("Input tokens:  %d\n", u.InputTokens)
	fmt.Printf("Output tokens: %d\n", u.OutputTokens)

	return nil
}

func printTurn(name string, result *agent.TurnResult) {
	fmt.Printf("--- %s ---\n", name)
	for _, msg := range result.Messages {
		switch msg.Kind {
		case model.MessageKindLLM:
			fmt.Printf("LLM: %s\n", truncate(firstText(msg), 110))
		case model.MessageKindCompaction:
			fmt.Printf("Compaction: %s\n", truncate(firstText(msg), 110))
		}
	}
	fmt.Println()
}

func firstText(msg model.Message) string {
	for _, p := range msg.Content {
		if p.Type == model.ContentPartTypeText && p.Text != "" {
			return p.Text
		}
	}

	return ""
}

func textParts(text string) []model.ContentPart {
	return []model.ContentPart{model.NewContentText(text)}
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}

	return s[:max] + "..."
}

func timeoutFromEnv(envVar string, def time.Duration) (time.Duration, error) {
	v := os.Getenv(envVar)
	if v == "" {
		return def, nil
	}

	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q: %w", envVar, v, err)
	}

	if d <= 0 {
		return 0, fmt.Errorf("invalid %s %q: must be > 0", envVar, v)
	}

	return d, nil
}
