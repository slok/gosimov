// Example pprof runs deterministic gosimov workloads for profiling.
//
// It is intentionally simple: no custom stats aggregation, only standard Go tooling.
//
// Usage:
//
//	go run ./examples/pprof
//	go run ./examples/pprof --mode tools --turns 5000 --cpu-profile cpu.pprof --heap-profile heap.pprof
//	go run ./examples/pprof --mode compaction --compact-every 20 --pprof-addr :6060
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	runtimepprof "runtime/pprof"
	"strings"
	"time"

	"github.com/slok/gosimov/internal/benchharness"
)

type config struct {
	mode              string
	sessions          int
	turns             int
	sessionResetEvery int
	promptBytes       int
	responseBytes     int
	toolCalls         int
	toolResultBytes   int
	compactEvery      int
	keepRecentTokens  int
	store             string
	storeDir          string

	pprofAddr        string
	cpuProfile       string
	heapProfile      string
	goroutineProfile string
	blockProfile     string
	mutexProfile     string
}

func loadConfig() (config, error) {
	var cfg config

	flag.StringVar(&cfg.mode, "mode", string(benchharness.ModeMixed), "Workload mode: simple|tools|compaction|mixed")
	flag.IntVar(&cfg.sessions, "sessions", 1, "Number of concurrent sessions")
	flag.IntVar(&cfg.turns, "turns", 2000, "Turns per session")
	flag.IntVar(&cfg.sessionResetEvery, "session-reset-every", 0, "Reset session every N turns (0 disables)")
	flag.IntVar(&cfg.promptBytes, "prompt-bytes", 256, "Prompt payload size in bytes")
	flag.IntVar(&cfg.responseBytes, "response-bytes", 256, "LLM response payload size in bytes")
	flag.IntVar(&cfg.toolCalls, "tool-calls", 0, "Tool calls per turn (0 uses mode default)")
	flag.IntVar(&cfg.toolResultBytes, "tool-result-bytes", 512, "Tool result payload size in bytes")
	flag.IntVar(&cfg.compactEvery, "compact-every", 0, "Force compact every N turns (0 disables, compaction mode defaults to 50)")
	flag.IntVar(&cfg.keepRecentTokens, "keep-recent-tokens", 1200, "Compactor keep recent tokens")
	flag.StringVar(&cfg.store, "store", string(benchharness.StoreMemory), "Store backend: memory|jsonl")
	flag.StringVar(&cfg.storeDir, "store-dir", "", "Store dir for jsonl (empty uses temp dir)")

	flag.StringVar(&cfg.pprofAddr, "pprof-addr", "", "Enable HTTP pprof server on this address (e.g. :6060)")
	flag.StringVar(&cfg.cpuProfile, "cpu-profile", "", "Write CPU profile to file")
	flag.StringVar(&cfg.heapProfile, "heap-profile", "", "Write heap profile at the end")
	flag.StringVar(&cfg.goroutineProfile, "goroutine-profile", "", "Write goroutine profile at the end")
	flag.StringVar(&cfg.blockProfile, "block-profile", "", "Write block profile at the end")
	flag.StringVar(&cfg.mutexProfile, "mutex-profile", "", "Write mutex profile at the end")

	flag.Parse()

	cfg.mode = strings.TrimSpace(cfg.mode)
	cfg.store = strings.TrimSpace(cfg.store)
	cfg.storeDir = strings.TrimSpace(cfg.storeDir)
	cfg.pprofAddr = strings.TrimSpace(cfg.pprofAddr)
	cfg.cpuProfile = strings.TrimSpace(cfg.cpuProfile)
	cfg.heapProfile = strings.TrimSpace(cfg.heapProfile)
	cfg.goroutineProfile = strings.TrimSpace(cfg.goroutineProfile)
	cfg.blockProfile = strings.TrimSpace(cfg.blockProfile)
	cfg.mutexProfile = strings.TrimSpace(cfg.mutexProfile)

	if cfg.sessions <= 0 {
		return config{}, fmt.Errorf("--sessions must be > 0")
	}

	if cfg.turns <= 0 {
		return config{}, fmt.Errorf("--turns must be > 0")
	}

	if cfg.promptBytes <= 0 {
		return config{}, fmt.Errorf("--prompt-bytes must be > 0")
	}

	if cfg.responseBytes <= 0 {
		return config{}, fmt.Errorf("--response-bytes must be > 0")
	}

	if cfg.toolResultBytes <= 0 {
		return config{}, fmt.Errorf("--tool-result-bytes must be > 0")
	}

	if cfg.toolCalls < 0 {
		return config{}, fmt.Errorf("--tool-calls must be >= 0")
	}

	if cfg.sessionResetEvery < 0 {
		return config{}, fmt.Errorf("--session-reset-every must be >= 0")
	}

	if cfg.compactEvery < 0 {
		return config{}, fmt.Errorf("--compact-every must be >= 0")
	}

	if cfg.keepRecentTokens <= 0 {
		return config{}, fmt.Errorf("--keep-recent-tokens must be > 0")
	}

	return cfg, nil
}

func run(ctx context.Context) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	if cfg.pprofAddr != "" {
		go func() {
			if err := http.ListenAndServe(cfg.pprofAddr, nil); err != nil {
				fmt.Fprintf(os.Stderr, "pprof server error: %v\n", err)
			}
		}()
	}

	if cfg.blockProfile != "" {
		runtime.SetBlockProfileRate(1)
	}

	if cfg.mutexProfile != "" {
		runtime.SetMutexProfileFraction(1)
	}

	stopCPU, err := maybeStartCPUProfile(cfg.cpuProfile)
	if err != nil {
		return err
	}
	defer stopCPU()

	start := time.Now()
	result, err := benchharness.Run(ctx, benchharness.Config{
		Mode:              benchharness.Mode(cfg.mode),
		Sessions:          cfg.sessions,
		Turns:             cfg.turns,
		SessionResetEvery: cfg.sessionResetEvery,
		PromptBytes:       cfg.promptBytes,
		ResponseBytes:     cfg.responseBytes,
		ToolCalls:         cfg.toolCalls,
		ToolResultBytes:   cfg.toolResultBytes,
		CompactEvery:      cfg.compactEvery,
		KeepRecentTokens:  cfg.keepRecentTokens,
		Store:             benchharness.StoreKind(cfg.store),
		StoreDir:          cfg.storeDir,
	})
	if err != nil {
		return err
	}

	elapsed := time.Since(start)

	if err := writeProfile("heap", cfg.heapProfile); err != nil {
		return err
	}

	if err := writeProfile("goroutine", cfg.goroutineProfile); err != nil {
		return err
	}

	if err := writeProfile("block", cfg.blockProfile); err != nil {
		return err
	}

	if err := writeProfile("mutex", cfg.mutexProfile); err != nil {
		return err
	}

	fmt.Printf("Mode:          %s\n", cfg.mode)
	fmt.Printf("Store:         %s\n", cfg.store)
	fmt.Printf("Sessions:      %d\n", result.Sessions)
	fmt.Printf("Turns:         %d\n", result.Turns)
	fmt.Printf("Elapsed:       %s\n", elapsed.Round(time.Millisecond))

	if cfg.pprofAddr != "" {
		fmt.Printf("HTTP pprof:    http://localhost%s/debug/pprof/\n", cfg.pprofAddr)
	}

	if cfg.cpuProfile != "" || cfg.heapProfile != "" || cfg.goroutineProfile != "" || cfg.blockProfile != "" || cfg.mutexProfile != "" {
		fmt.Println("Profiles:      written")
	}

	return nil
}

func maybeStartCPUProfile(path string) (func(), error) {
	if path == "" {
		return func() {}, nil
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("creating cpu profile file: %w", err)
	}

	if err := runtimepprof.StartCPUProfile(f); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("starting cpu profile: %w", err)
	}

	return func() {
		runtimepprof.StopCPUProfile()
		_ = f.Close()
	}, nil
}

func writeProfile(name, path string) error {
	if path == "" {
		return nil
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s profile file: %w", name, err)
	}
	defer f.Close()

	if name == "heap" {
		runtime.GC()
	}

	p := runtimepprof.Lookup(name)
	if p == nil {
		return fmt.Errorf("profile %q not available", name)
	}

	if err := p.WriteTo(f, 0); err != nil {
		return fmt.Errorf("writing %s profile: %w", name, err)
	}

	return nil
}

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}
