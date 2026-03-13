package agent

import (
	"context"

	"github.com/slok/gosimov/pkg/model"
)

type ctxKey string

const (
	ctxKeySessionID    ctxKey = "agent-session-id"
	ctxKeyLLMModelInfo ctxKey = "agent-llm-model-info"
)

// CtxWithSessionID returns a copy of parent with the session ID set.
func ctxWithSessionID(parent context.Context, sessionID string) context.Context {
	return context.WithValue(parent, ctxKeySessionID, sessionID)
}

// SessionIDFromCtx gets the session ID from context.
func SessionIDFromCtx(ctx context.Context) string {
	sessionID, _ := ctx.Value(ctxKeySessionID).(string)
	return sessionID
}

// CtxWithLLMModelInfo returns a copy of parent with the model info set.
// A nil model info is ignored and parent is returned as-is.
func ctxWithLLMModelInfo(parent context.Context, info *model.LLMModelInfo) context.Context {
	if info == nil {
		return parent
	}

	cpy := *info

	return context.WithValue(parent, ctxKeyLLMModelInfo, &cpy)
}

// LLMModelInfoFromCtx gets the model info from context.
func LLMModelInfoFromCtx(ctx context.Context) *model.LLMModelInfo {
	info, _ := ctx.Value(ctxKeyLLMModelInfo).(*model.LLMModelInfo)
	return info
}
