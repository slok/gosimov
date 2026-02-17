package model

// LLMModelInputModality is a known input modality for an LLM model.
type LLMModelInputModality string

const (
	LLMModelInputModalityText  LLMModelInputModality = "text"
	LLMModelInputModalityImage LLMModelInputModality = "image"
	LLMModelInputModalityAudio LLMModelInputModality = "audio"
	LLMModelInputModalityVideo LLMModelInputModality = "video"
	LLMModelInputModalityPDF   LLMModelInputModality = "pdf"
)

// LLMModelOutputModality is a known output modality for an LLM model.
type LLMModelOutputModality string

const (
	LLMModelOutputModalityText  LLMModelOutputModality = "text"
	LLMModelOutputModalityImage LLMModelOutputModality = "image"
	LLMModelOutputModalityAudio LLMModelOutputModality = "audio"
	LLMModelOutputModalityVideo LLMModelOutputModality = "video"
)

// LLMModelInfo contains normalized model capabilities metadata.
type LLMModelInfo struct {
	ID               string
	Name             string
	Reasoning        bool
	ToolCall         bool
	ContextWindow    int
	MaxOutputTokens  int
	InputModalities  []LLMModelInputModality
	OutputModalities []LLMModelOutputModality
}
