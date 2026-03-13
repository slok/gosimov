package log

// KV is a helper type for structured fields.
type KV = map[string]any

// Logger is the minimal logging interface used by the project.
type Logger interface {
	Infof(format string, args ...any)
	Debugf(format string, args ...any)
	WithValues(values KV) Logger
}

// Noop logger doesn't log anything.
const Noop = noop(0)

type noop int

func (n noop) Infof(_ string, _ ...any)  {}
func (n noop) Debugf(_ string, _ ...any) {}
func (n noop) WithValues(_ KV) Logger    { return n }
