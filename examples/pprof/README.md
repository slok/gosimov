# pprof example

This example runs deterministic gosimov workloads with a fake provider so you can profile SDK behavior without provider/network noise.

It does not aggregate custom benchmark metrics. Use Go standard tooling (`pprof`, `go test -bench`) for analysis.

## Run

```bash
# Default mixed workload.
go run ./examples/pprof

# Tool-heavy workload + profile files.
go run ./examples/pprof --mode tools --turns 5000 --cpu-profile cpu.pprof --heap-profile heap.pprof

# Compaction-heavy workload + live pprof endpoint.
go run ./examples/pprof --mode compaction --compact-every 20 --pprof-addr :6060
```

## Useful flags

- `--mode`: `simple|tools|compaction|mixed`
- `--sessions`: concurrent sessions
- `--turns`: turns per session
- `--session-reset-every`: recreate session every N turns
- `--tool-calls`: tool rounds per turn (`0` uses mode defaults)
- `--compact-every`: force compact every N turns
- `--store`: `memory|jsonl`
- `--store-dir`: jsonl directory (empty uses a temp dir)

Profiles:

- `--pprof-addr`: live HTTP pprof endpoint
- `--cpu-profile`
- `--heap-profile`
- `--goroutine-profile`
- `--block-profile`
- `--mutex-profile`

## Benchmark workflow (Go standard tooling)

Use the benchmark harness to compare modes and allocations:

```bash
go test ./tests/benchmark -run '^$' -bench BenchmarkSessionHarness -benchmem -benchtime=2s
```

Capture benchmark profiles:

```bash
go test ./tests/benchmark \
  -run '^$' \
  -bench 'BenchmarkSessionHarness/tools_memory' \
  -benchmem \
  -benchtime=5s \
  -cpuprofile /tmp/gosimov-bench-tools-memory.cpu.pprof \
  -memprofile /tmp/gosimov-bench-tools-memory.mem.pprof
```

Inspect them:

```bash
go tool pprof -top /tmp/gosimov-bench-tools-memory.cpu.pprof
go tool pprof -top -sample_index=alloc_space /tmp/gosimov-bench-tools-memory.mem.pprof
go tool pprof -web /tmp/gosimov-bench-tools-memory.cpu.pprof
```

## Example run workflow (pprof files)

```bash
go run ./examples/pprof \
  --mode mixed \
  --sessions 2 \
  --turns 3000 \
  --cpu-profile /tmp/gosimov-example-mixed.cpu.pprof \
  --heap-profile /tmp/gosimov-example-mixed.heap.pprof \
  --goroutine-profile /tmp/gosimov-example-mixed.goroutine.pprof

go tool pprof -top /tmp/gosimov-example-mixed.cpu.pprof
go tool pprof -top -sample_index=alloc_space /tmp/gosimov-example-mixed.heap.pprof
go tool pprof -top /tmp/gosimov-example-mixed.goroutine.pprof
```

## What to look for first

- `pkg/agent.runTurn`: often dominates alloc-space in tool-heavy runs.
- `pkg/agent/context/simple.*`: `checkpointAndFollowing`, `copyMessages`, and lookup helpers can show up in CPU/alloc profiles.
- GC/runtime pressure: high `runtime.scanObject`, `runtime.tryDeferToSpanScan`, and write-barrier frames usually indicate allocation churn.

This harness is intentionally synthetic and deterministic. Use it for relative comparisons between branches and changes, then confirm with real integration flows.
