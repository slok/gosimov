package customopenaicompatible

type qwenThinkingMode string

const (
	qwenThinkingModeNone         qwenThinkingMode = ""
	qwenThinkingModeTopLevel     qwenThinkingMode = "top-level"
	qwenThinkingModeChatTemplate qwenThinkingMode = "chat-template"
)

// ProviderOptions configures provider-specific request customizations.
type ProviderOptions struct {
	qwenThinkingEnabled  *bool
	qwenThinkingMode     qwenThinkingMode
	qwenPreserveThinking *bool
	rawRequestFields     map[string]any
}

// NewProviderOptions creates empty provider-specific options.
func NewProviderOptions() ProviderOptions {
	return ProviderOptions{}
}

// WithQwenThinking configures Qwen-compatible providers that expect
// `enable_thinking` at the top level of the request body.
func (o ProviderOptions) WithQwenThinking(enabled bool) ProviderOptions {
	o.qwenThinkingEnabled = &enabled
	o.qwenThinkingMode = qwenThinkingModeTopLevel

	return o
}

// WithQwenChatTemplateThinking configures Qwen-compatible providers that expect
// `chat_template_kwargs.enable_thinking` in the request body.
func (o ProviderOptions) WithQwenChatTemplateThinking(enabled bool) ProviderOptions {
	o.qwenThinkingEnabled = &enabled
	o.qwenThinkingMode = qwenThinkingModeChatTemplate

	return o
}

// WithQwenChatTemplatePreserveThinking configures Qwen-compatible providers that
// expect `chat_template_kwargs.preserve_thinking` in the request body.
func (o ProviderOptions) WithQwenChatTemplatePreserveThinking(enabled bool) ProviderOptions {
	o.qwenPreserveThinking = &enabled
	if o.qwenThinkingMode == qwenThinkingModeNone {
		o.qwenThinkingMode = qwenThinkingModeChatTemplate
	}

	return o
}

// WithRawRequestField adds a raw request-body field for unsupported provider
// quirks. Prefer the typed helpers when available.
func (o ProviderOptions) WithRawRequestField(key string, value any) ProviderOptions {
	o.rawRequestFields = cloneMap(o.rawRequestFields)
	if o.rawRequestFields == nil {
		o.rawRequestFields = map[string]any{}
	}
	o.rawRequestFields[key] = value

	return o
}

func (o ProviderOptions) requestBodyFields() map[string]any {
	result := cloneMap(o.rawRequestFields)
	if result == nil {
		result = map[string]any{}
	}

	switch o.qwenThinkingMode {
	case qwenThinkingModeTopLevel:
		if o.qwenThinkingEnabled != nil {
			result["enable_thinking"] = *o.qwenThinkingEnabled
		}
	case qwenThinkingModeChatTemplate:
		kwargs := map[string]any{}
		if existing, ok := result["chat_template_kwargs"].(map[string]any); ok {
			kwargs = cloneMap(existing)
		}
		if o.qwenThinkingEnabled != nil {
			kwargs["enable_thinking"] = *o.qwenThinkingEnabled
		}
		if o.qwenPreserveThinking != nil {
			kwargs["preserve_thinking"] = *o.qwenPreserveThinking
		}
		if len(kwargs) > 0 {
			result["chat_template_kwargs"] = kwargs
		}
	}

	if len(result) == 0 {
		return nil
	}

	return result
}

func cloneMap(src map[string]any) map[string]any {
	if len(src) == 0 {
		return nil
	}

	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}

	return dst
}
