package zero

import (
	"github.com/rs/zerolog"

	"github.com/slok/gosimov/pkg/log"
)

type logger struct {
	l zerolog.Logger
}

// New returns a log.Logger backed by zerolog.
func New(l zerolog.Logger) log.Logger {
	return logger{l: l}
}

func (l logger) Infof(format string, args ...any) {
	l.l.Info().Msgf(format, args...)
}

func (l logger) Debugf(format string, args ...any) {
	l.l.Debug().Msgf(format, args...)
}

func (l logger) WithValues(kv log.KV) log.Logger {
	ctx := l.l.With()
	for k, v := range kv {
		ctx = ctx.Interface(k, v)
	}

	return New(ctx.Logger())
}
