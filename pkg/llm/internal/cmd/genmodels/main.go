package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"text/template"
)

const (
	modelsDevURL = "https://models.dev/api.json"

	providerIDOpenAI     = "openai"
	providerIDZen        = "opencode"
	providerIDAnthropic  = "anthropic"
	providerIDOpenCodeGo = "opencode-go"

	providerNPMAnthropic = "@ai-sdk/anthropic"

	modelIDMinimaxM27 = "minimax-m2.7"
	modelIDQwen35Plus = "qwen3.5-plus"
	modelIDQwen36Plus = "qwen3.6-plus"

	goModelAPIFormatOpenAICompatible = "modelAPIFormatOpenAICompatible"
	goModelAPIFormatAnthropic        = "modelAPIFormatAnthropic"
)

type providerData struct {
	NPM    string               `json:"npm"`
	Models map[string]modelData `json:"models"`
}

type modelData struct {
	ID         string       `json:"id"`
	Name       string       `json:"name"`
	Reasoning  bool         `json:"reasoning"`
	ToolCall   bool         `json:"tool_call"`
	Modalities modalities   `json:"modalities"`
	Limit      limitData    `json:"limit"`
	Status     string       `json:"status"`
	Provider   *modelSource `json:"provider,omitempty"`
}

type modelSource struct {
	NPM string `json:"npm"`
}

type modalities struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

type limitData struct {
	Context int `json:"context"`
	Output  int `json:"output"`
}

type entry struct {
	ID               string
	ConstName        string
	Name             string
	ProviderNPM      string
	APIFormat        string
	Reasoning        bool
	ToolCall         bool
	ContextWindow    int
	MaxOutputTokens  int
	InputModalities  []string
	OutputModalities []string
	InputConsts      []string
	OutputConsts     []string
}

type openAITemplateData struct {
	Entries        []entry
	ChatGPTEntries []entry
}

type zenTemplateData struct {
	Entries []entry
}

type anthropicTemplateData struct {
	Entries []entry
}

type opencodeGoTemplateData struct {
	Entries []entry
}

func main() {
	target := flag.String("target", "", "catalog target: openai, zen, anthropic or opencode-go")
	out := flag.String("out", "", "output go file path")
	flag.Parse()

	if strings.TrimSpace(*target) == "" || strings.TrimSpace(*out) == "" {
		fatalf("-target and -out are required")
	}

	raw, err := fetchModelsDev(modelsDevURL)
	if err != nil {
		fatalf("fetching models.dev: %v", err)
	}

	providers := map[string]providerData{}
	if err := json.Unmarshal(raw, &providers); err != nil {
		fatalf("decoding models.dev payload: %v", err)
	}

	var generated []byte
	switch *target {
	case providerIDOpenAI:
		generated, err = generateOpenAI(providers)
	case "zen":
		generated, err = generateZen(providers)
	case providerIDAnthropic:
		generated, err = generateAnthropic(providers)
	case providerIDOpenCodeGo:
		generated, err = generateOpenCodeGo(providers)
	default:
		fatalf("unsupported target %q", *target)
	}
	if err != nil {
		fatalf("generating %s catalog: %v", *target, err)
	}

	if err := os.WriteFile(*out, generated, 0o644); err != nil {
		fatalf("writing output: %v", err)
	}
}

func generateOpenAI(providers map[string]providerData) ([]byte, error) {
	src, ok := providers[providerIDOpenAI]
	if !ok {
		return nil, fmt.Errorf("provider openai not found")
	}

	all := normalizeEntries(providerIDOpenAI, src.Models, src.NPM, func(modelData) bool { return true })
	chatgpt := normalizeEntries(providerIDOpenAI, src.Models, src.NPM, isChatGPTModel)

	return renderTemplate(openAITemplate, openAITemplateData{Entries: all, ChatGPTEntries: chatgpt})
}

func generateZen(providers map[string]providerData) ([]byte, error) {
	src, ok := providers[providerIDZen]
	if !ok {
		return nil, fmt.Errorf("provider opencode not found")
	}

	entries := normalizeEntries(providerIDZen, src.Models, src.NPM, func(modelData) bool { return true })

	return renderTemplate(zenTemplate, zenTemplateData{Entries: entries})
}

func generateAnthropic(providers map[string]providerData) ([]byte, error) {
	src, ok := providers[providerIDAnthropic]
	if !ok {
		return nil, fmt.Errorf("provider anthropic not found")
	}

	entries := normalizeEntries(providerIDAnthropic, src.Models, src.NPM, func(modelData) bool { return true })

	return renderTemplate(anthropicTemplate, anthropicTemplateData{Entries: entries})
}

func generateOpenCodeGo(providers map[string]providerData) ([]byte, error) {
	src, ok := providers[providerIDOpenCodeGo]
	if !ok {
		return nil, fmt.Errorf("provider opencode-go not found")
	}

	entries := normalizeEntries(providerIDOpenCodeGo, src.Models, src.NPM, func(modelData) bool { return true })

	return renderTemplate(opencodeGoTemplate, opencodeGoTemplateData{Entries: entries})
}
func renderTemplate(raw string, data any) ([]byte, error) {
	tmpl, err := template.New("models").Funcs(template.FuncMap{"joinSymbols": joinSymbols}).Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("parsing template: %w", err)
	}

	var b bytes.Buffer
	if err := tmpl.Execute(&b, data); err != nil {
		return nil, fmt.Errorf("executing template: %w", err)
	}

	formatted, err := format.Source(b.Bytes())
	if err != nil {
		return nil, fmt.Errorf("formatting generated source: %w", err)
	}

	return formatted, nil
}

func joinSymbols(xs []string) string {
	if len(xs) == 0 {
		return ""
	}
	return strings.Join(xs, ", ")
}

const openAITemplate = `// Code generated by pkg/llm/internal/cmd/genmodels. DO NOT EDIT.
// Source: https://models.dev/api.json

package openai

import "github.com/slok/gosimov/pkg/model"

const (
{{- range .Entries }}
	{{ .ConstName }} = {{ printf "%q" .ID }}
{{- end }}
)

var openAIModelIDs = []string{
{{- range .Entries }}
	{{ printf "%q" .ID }},
{{- end }}
}

var openAIModelsByID = map[string]model.LLMModelInfo{
{{- range .Entries }}
	{{ printf "%q" .ID }}: {
		ID:              {{ printf "%q" .ID }},
		Name:            {{ printf "%q" .Name }},
		Reasoning:       {{ .Reasoning }},
		ToolCall:        {{ .ToolCall }},
		ContextWindow:   {{ .ContextWindow }},
		MaxOutputTokens: {{ .MaxOutputTokens }},
		InputModalities: []model.LLMModelInputModality{ {{ joinSymbols .InputConsts }} },
		OutputModalities: []model.LLMModelOutputModality{ {{ joinSymbols .OutputConsts }} },
	},
{{- end }}
}

var chatGPTModelIDs = []string{
{{- range .ChatGPTEntries }}
	{{ printf "%q" .ID }},
{{- end }}
}

var chatGPTModelsByID = map[string]model.LLMModelInfo{
{{- range .ChatGPTEntries }}
	{{ printf "%q" .ID }}: {
		ID:              {{ printf "%q" .ID }},
		Name:            {{ printf "%q" .Name }},
		Reasoning:       {{ .Reasoning }},
		ToolCall:        {{ .ToolCall }},
		ContextWindow:   {{ .ContextWindow }},
		MaxOutputTokens: {{ .MaxOutputTokens }},
		InputModalities: []model.LLMModelInputModality{ {{ joinSymbols .InputConsts }} },
		OutputModalities: []model.LLMModelOutputModality{ {{ joinSymbols .OutputConsts }} },
	},
{{- end }}
}

// IsSupportedOpenAIModel returns true if model ID exists in the OpenAI catalog.
func IsSupportedOpenAIModel(modelID string) bool {
	_, ok := openAIModelsByID[modelID]
	return ok
}

// IsSupportedChatGPTModel returns true if model ID exists in the ChatGPT catalog.
func IsSupportedChatGPTModel(modelID string) bool {
	_, ok := chatGPTModelsByID[modelID]
	return ok
}

// OpenAIModelInfo returns OpenAI model metadata when present.
func OpenAIModelInfo(modelID string) (model.LLMModelInfo, bool) {
	v, ok := openAIModelsByID[modelID]
	return v, ok
}

// ChatGPTModelInfo returns ChatGPT model metadata when present.
func ChatGPTModelInfo(modelID string) (model.LLMModelInfo, bool) {
	v, ok := chatGPTModelsByID[modelID]
	return v, ok
}

// SupportedOpenAIModelIDs returns all known OpenAI model IDs.
func SupportedOpenAIModelIDs() []string {
	ids := make([]string, len(openAIModelIDs))
	copy(ids, openAIModelIDs)
	return ids
}

// SupportedChatGPTModelIDs returns all known ChatGPT model IDs.
func SupportedChatGPTModelIDs() []string {
	ids := make([]string, len(chatGPTModelIDs))
	copy(ids, chatGPTModelIDs)
	return ids
}
`

const zenTemplate = `// Code generated by pkg/llm/internal/cmd/genmodels. DO NOT EDIT.
// Source: https://models.dev/api.json

package zen

import "github.com/slok/gosimov/pkg/model"

const (
{{- range .Entries }}
	{{ .ConstName }} = {{ printf "%q" .ID }}
{{- end }}
)

var modelIDs = []string{
{{- range .Entries }}
	{{ printf "%q" .ID }},
{{- end }}
}

var modelsByID = map[string]model.LLMModelInfo{
{{- range .Entries }}
	{{ printf "%q" .ID }}: {
		ID:              {{ printf "%q" .ID }},
		Name:            {{ printf "%q" .Name }},
		Reasoning:       {{ .Reasoning }},
		ToolCall:        {{ .ToolCall }},
		ContextWindow:   {{ .ContextWindow }},
		MaxOutputTokens: {{ .MaxOutputTokens }},
		InputModalities: []model.LLMModelInputModality{ {{ joinSymbols .InputConsts }} },
		OutputModalities: []model.LLMModelOutputModality{ {{ joinSymbols .OutputConsts }} },
	},
{{- end }}
}

// IsSupportedModel returns true if model ID exists in the Zen catalog.
func IsSupportedModel(modelID string) bool {
	_, ok := modelsByID[modelID]
	return ok
}

// ModelByID returns model metadata when present.
func ModelByID(modelID string) (model.LLMModelInfo, bool) {
	v, ok := modelsByID[modelID]
	return v, ok
}

// SupportedModelIDs returns all known Zen model IDs.
func SupportedModelIDs() []string {
	ids := make([]string, len(modelIDs))
	copy(ids, modelIDs)
	return ids
}
`

const anthropicTemplate = `// Code generated by pkg/llm/internal/cmd/genmodels. DO NOT EDIT.
// Source: https://models.dev/api.json

package anthropic

import "github.com/slok/gosimov/pkg/model"

const (
{{- range .Entries }}
	{{ .ConstName }} = {{ printf "%q" .ID }}
{{- end }}
)

var modelIDs = []string{
{{- range .Entries }}
	{{ printf "%q" .ID }},
{{- end }}
}

var modelsByID = map[string]model.LLMModelInfo{
{{- range .Entries }}
	{{ printf "%q" .ID }}: {
		ID:              {{ printf "%q" .ID }},
		Name:            {{ printf "%q" .Name }},
		Reasoning:       {{ .Reasoning }},
		ToolCall:        {{ .ToolCall }},
		ContextWindow:   {{ .ContextWindow }},
		MaxOutputTokens: {{ .MaxOutputTokens }},
		InputModalities: []model.LLMModelInputModality{ {{ joinSymbols .InputConsts }} },
		OutputModalities: []model.LLMModelOutputModality{ {{ joinSymbols .OutputConsts }} },
	},
{{- end }}
}

// IsSupportedModel returns true if model ID exists in the Anthropic catalog.
func IsSupportedModel(modelID string) bool {
	_, ok := modelsByID[modelID]
	return ok
}

// ModelByID returns model metadata when present.
func ModelByID(modelID string) (model.LLMModelInfo, bool) {
	v, ok := modelsByID[modelID]
	return v, ok
}

// SupportedModelIDs returns all known Anthropic model IDs.
func SupportedModelIDs() []string {
	ids := make([]string, len(modelIDs))
	copy(ids, modelIDs)
	return ids
}
`

const opencodeGoTemplate = `// Code generated by pkg/llm/internal/cmd/genmodels. DO NOT EDIT.
// Source: https://models.dev/api.json

package opencodego

import "github.com/slok/gosimov/pkg/model"

type modelAPIFormat string

const (
	modelAPIFormatOpenAICompatible modelAPIFormat = "openai-compatible"
	modelAPIFormatAnthropic        modelAPIFormat = "anthropic"
)

const (
{{- range .Entries }}
	{{ .ConstName }} = {{ printf "%q" .ID }}
{{- end }}
)

var modelIDs = []string{
{{- range .Entries }}
	{{ printf "%q" .ID }},
{{- end }}
}

var modelsByID = map[string]model.LLMModelInfo{
{{- range .Entries }}
	{{ printf "%q" .ID }}: {
		ID:              {{ printf "%q" .ID }},
		Name:            {{ printf "%q" .Name }},
		Reasoning:       {{ .Reasoning }},
		ToolCall:        {{ .ToolCall }},
		ContextWindow:   {{ .ContextWindow }},
		MaxOutputTokens: {{ .MaxOutputTokens }},
		InputModalities: []model.LLMModelInputModality{ {{ joinSymbols .InputConsts }} },
		OutputModalities: []model.LLMModelOutputModality{ {{ joinSymbols .OutputConsts }} },
	},
{{- end }}
}

var modelFormatsByID = map[string]modelAPIFormat{
{{- range .Entries }}
	// ProviderNPM comes from models.dev (provider-level npm or model-level override).
	// We use it as the canonical signal for the underlying API shape this model expects.
	// - @ai-sdk/anthropic         -> Anthropic Messages API shape (/messages)
	// - everything else for Go now -> OpenAI-compatible Chat API shape (/chat/completions)
	{{ printf "%q" .ID }}: {{ .APIFormat }},
{{- end }}
}

// IsSupportedModel returns true if model ID exists in the OpenCode Go catalog.
func IsSupportedModel(modelID string) bool {
	_, ok := modelsByID[modelID]
	return ok
}

// ModelByID returns model metadata when present.
func ModelByID(modelID string) (model.LLMModelInfo, bool) {
	v, ok := modelsByID[modelID]
	return v, ok
}

// ModelFormatByID returns API format metadata when present.
func ModelFormatByID(modelID string) (modelAPIFormat, bool) {
	v, ok := modelFormatsByID[modelID]
	return v, ok
}

// SupportedModelIDs returns all known OpenCode Go model IDs.
func SupportedModelIDs() []string {
	ids := make([]string, len(modelIDs))
	copy(ids, modelIDs)
	return ids
}
`

func normalizeEntries(providerID string, models map[string]modelData, defaultProviderNPM string, include func(modelData) bool) []entry {
	ids := make([]string, 0, len(models))
	for id := range models {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	result := make([]entry, 0, len(ids))
	seenConst := map[string]int{}
	for _, id := range ids {
		m := models[id]
		if !include(m) {
			continue
		}

		constName := constName(m.ID)
		if c := seenConst[constName]; c > 0 {
			constName = fmt.Sprintf("%s_%d", constName, c+1)
		}
		seenConst[constName]++

		result = append(result, entry{
			ID:               m.ID,
			ConstName:        constName,
			Name:             m.Name,
			ProviderNPM:      modelProviderNPM(m, defaultProviderNPM),
			APIFormat:        modelAPIFormatConst(providerID, defaultProviderNPM, m),
			Reasoning:        m.Reasoning,
			ToolCall:         m.ToolCall,
			ContextWindow:    m.Limit.Context,
			MaxOutputTokens:  m.Limit.Output,
			InputModalities:  sortedCopy(m.Modalities.Input),
			OutputModalities: sortedCopy(m.Modalities.Output),
			InputConsts:      inputModalityConsts(sortedCopy(m.Modalities.Input)),
			OutputConsts:     outputModalityConsts(sortedCopy(m.Modalities.Output)),
		})
	}

	return result
}

func modelProviderNPM(m modelData, defaultProviderNPM string) string {
	// models.dev can define npm at provider level and optionally override per model.
	// This npm value indicates the SDK/provider implementation kind (API contract shape),
	// not just a package name for installation purposes.
	if m.Provider != nil && strings.TrimSpace(m.Provider.NPM) != "" {
		return strings.TrimSpace(m.Provider.NPM)
	}

	return strings.TrimSpace(defaultProviderNPM)
}

func modelAPIFormatConst(providerID, defaultProviderNPM string, m modelData) string {
	npm := modelProviderNPM(m, defaultProviderNPM)
	if providerID == providerIDOpenCodeGo && opencodeGoOpenAICompatibleModel(m.ID) {
		return goModelAPIFormatOpenAICompatible
	}

	if npm == providerNPMAnthropic {
		return goModelAPIFormatAnthropic
	}

	return goModelAPIFormatOpenAICompatible
}

func opencodeGoOpenAICompatibleModel(modelID string) bool {
	switch modelID {
	case modelIDMinimaxM27, modelIDQwen35Plus, modelIDQwen36Plus:
		// models.dev marks these OpenCode Go models as Anthropic-backed, but the
		// actual OpenCode Go endpoint expects the OpenAI-compatible chat route.
		return true
	default:
		return false
	}
}

func inputModalityConsts(mods []string) []string {
	out := make([]string, 0, len(mods))
	for _, m := range mods {
		switch m {
		case "text":
			out = append(out, "model.LLMModelInputModalityText")
		case "image":
			out = append(out, "model.LLMModelInputModalityImage")
		case "audio":
			out = append(out, "model.LLMModelInputModalityAudio")
		case "video":
			out = append(out, "model.LLMModelInputModalityVideo")
		case "pdf":
			out = append(out, "model.LLMModelInputModalityPDF")
		default:
			out = append(out, fmt.Sprintf("model.LLMModelInputModality(%q)", m))
		}
	}

	return out
}

func outputModalityConsts(mods []string) []string {
	out := make([]string, 0, len(mods))
	for _, m := range mods {
		switch m {
		case "text":
			out = append(out, "model.LLMModelOutputModalityText")
		case "image":
			out = append(out, "model.LLMModelOutputModalityImage")
		case "audio":
			out = append(out, "model.LLMModelOutputModalityAudio")
		case "video":
			out = append(out, "model.LLMModelOutputModalityVideo")
		default:
			out = append(out, fmt.Sprintf("model.LLMModelOutputModality(%q)", m))
		}
	}

	return out
}

func isChatGPTModel(m modelData) bool {
	if m.Status == "deprecated" {
		return false
	}
	id := strings.ToLower(m.ID)
	return strings.Contains(id, "codex")
}

func constName(modelID string) string {
	var b strings.Builder
	b.WriteString("Model")
	upperNext := true
	for _, r := range modelID {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			if upperNext {
				if r >= 'a' && r <= 'z' {
					r = r - ('a' - 'A')
				}
				upperNext = false
			}
			b.WriteRune(r)
			continue
		}
		upperNext = true
	}

	return b.String()
}

func sortedCopy(xs []string) []string {
	out := append([]string(nil), xs...)
	sort.Strings(out)
	return out
}

func fetchModelsDev(url string) ([]byte, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "gosimov-genmodels/1.0")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	return body, nil
}

func fatalf(format string, args ...any) {
	_, _ = fmt.Fprintf(os.Stderr, "Error: "+format+"\n", args...)
	os.Exit(1)
}
