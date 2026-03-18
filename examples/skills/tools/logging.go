package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/slok/gosimov/pkg/tool"
)

type loggingTool struct {
	inner tool.Tool
}

func WrapWithLogging(inner tool.Tool, enabled bool) tool.Tool {
	if !enabled || inner == nil {
		return inner
	}

	return &loggingTool{inner: inner}
}

func (t *loggingTool) ID() string {
	return t.inner.ID()
}

func (t *loggingTool) Description() string {
	return t.inner.Description()
}

func (t *loggingTool) Schema() json.RawMessage {
	return t.inner.Schema()
}

func (t *loggingTool) Execute(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
	start := time.Now()
	res, err := t.inner.Execute(ctx, args)
	duration := time.Since(start).Round(time.Millisecond)
	argDetails := summarizeToolArgs(t.inner.ID(), args)

	if err != nil {
		log.Printf("[skills-example] tool=%s status=error duration=%s args_bytes=%d%s err=%s", t.inner.ID(), duration, len(args), argDetails, err)
		return nil, err
	}

	parts := 0
	if res != nil {
		parts = len(res.Content)
	}

	log.Printf("[skills-example] tool=%s status=ok duration=%s args_bytes=%d%s content_parts=%d", t.inner.ID(), duration, len(args), argDetails, parts)

	return res, nil
}

func summarizeToolArgs(toolID string, args json.RawMessage) string {
	switch toolID {
	case "shell":
		var in struct {
			Command string `json:"command"`
		}
		if err := json.Unmarshal(args, &in); err != nil || strings.TrimSpace(in.Command) == "" {
			return ""
		}

		return fmt.Sprintf(" command=%q", truncateArg(strings.TrimSpace(in.Command), 140))

	case "read":
		var in struct {
			Path   string `json:"path"`
			Offset int    `json:"offset"`
			Limit  int    `json:"limit"`
		}
		if err := json.Unmarshal(args, &in); err != nil || strings.TrimSpace(in.Path) == "" {
			return ""
		}

		detail := fmt.Sprintf(" path=%q", strings.TrimSpace(in.Path))
		if in.Offset > 0 {
			detail += fmt.Sprintf(" offset=%d", in.Offset)
		}
		if in.Limit > 0 {
			detail += fmt.Sprintf(" limit=%d", in.Limit)
		}

		return detail
	}

	return ""
}

func truncateArg(v string, max int) string {
	v = strings.ReplaceAll(v, "\n", "\\n")
	if len(v) <= max {
		return v
	}

	return v[:max] + "..."
}
