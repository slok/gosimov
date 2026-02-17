package pkgerrors

import "errors"

// Shared sentinel errors for programmatic error checking via [errors.Is].

var (
	// ErrNotValid indicates invalid input (missing required fields, bad config, etc.).
	ErrNotValid = errors.New("not valid")
	// ErrSessionBusy indicates a session operation was rejected because another is already running.
	ErrSessionBusy = errors.New("session is busy")
	// ErrMaxIterations indicates the agent loop exceeded its configured iteration limit.
	ErrMaxIterations = errors.New("max iterations exceeded")
	// ErrLLMError indicates the LLM returned an error stop reason in its response.
	ErrLLMError = errors.New("llm error")
	// ErrAborted indicates the operation was aborted (context cancelled, user abort, timeout).
	ErrAborted = errors.New("aborted")
	// ErrNotFound indicates the requested resource does not exist.
	ErrNotFound = errors.New("not found")
	// ErrAlreadyExists indicates the resource already exists (duplicate creation).
	ErrAlreadyExists = errors.New("already exists")
)
