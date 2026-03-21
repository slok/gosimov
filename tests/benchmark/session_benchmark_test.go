package benchmark_test

import (
	"context"
	"testing"

	"github.com/slok/gosimov/internal/benchharness"
)

func BenchmarkSessionHarness(b *testing.B) {
	b.ReportAllocs()

	benchmarks := map[string]benchharness.Config{
		"simple_memory": {
			Mode:        benchharness.ModeSimple,
			Sessions:    1,
			PromptBytes: 256,
			Store:       benchharness.StoreMemory,
		},
		"tools_memory": {
			Mode:            benchharness.ModeTools,
			Sessions:        1,
			PromptBytes:     256,
			ToolCalls:       2,
			ToolResultBytes: 512,
			Store:           benchharness.StoreMemory,
		},
		"compaction_memory": {
			Mode:             benchharness.ModeCompaction,
			Sessions:         1,
			PromptBytes:      256,
			ToolCalls:        2,
			ToolResultBytes:  512,
			CompactEvery:     20,
			KeepRecentTokens: 1200,
			Store:            benchharness.StoreMemory,
		},
		"tools_jsonl": {
			Mode:            benchharness.ModeTools,
			Sessions:        1,
			PromptBytes:     256,
			ToolCalls:       2,
			ToolResultBytes: 512,
			Store:           benchharness.StoreJSONL,
		},
	}

	for name, baseCfg := range benchmarks {
		b.Run(name, func(b *testing.B) {
			cfg := baseCfg
			cfg.Turns = b.N
			if cfg.Store == benchharness.StoreJSONL {
				cfg.StoreDir = b.TempDir()
			}

			b.ResetTimer()
			_, err := benchharness.Run(context.Background(), cfg)
			b.StopTimer()
			if err != nil {
				b.Fatalf("running benchmark harness: %v", err)
			}
		})
	}
}
