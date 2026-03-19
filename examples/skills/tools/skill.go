package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"text/template"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

const noSkillsDescription = "Load a specialized skill that provides domain-specific instructions and workflows. No skills are currently available."

type SkillToolConfig struct {
	// SkillsDir is the local skills directory when FS is not set.
	//
	// When FS is set, SkillsDir is used only as a display prefix in locations.
	SkillsDir string
	// FS is an optional skill filesystem. If set, skills are loaded from this FS.
	FS fs.FS
}

type skillTool struct {
	description string
	skills      []skillDefinition
	byName      map[string]skillDefinition
}

type skillDefinition struct {
	Name        string
	Description string
	Body        string
	Location    string
	BaseDir     string
}

type skillFrontmatter struct {
	Name        string
	Description string
}

type skillInput struct {
	Name string `json:"name" jsonschema:"required,description=The name of the skill from available_skills"`
}

var (
	skillSchema = toolschema.MustFromType[skillInput]()
	xmlEscaper  = strings.NewReplacer(
		"&", "&amp;",
		"<", "&lt;",
		">", "&gt;",
		`"`, "&quot;",
		"'", "&apos;",
	)
	funcMap = template.FuncMap{"xml": escapeXML}

	catalogTemplate = template.Must(template.New("skill-catalog").Funcs(funcMap).Parse(`Load a specialized skill that provides domain-specific instructions and workflows.

When you recognize that a task matches one of the skills listed below, call this tool with that skill name.

<available_skills>
{{- range .Skills }}
  <skill>
    <name>{{xml .Name}}</name>
    <description>{{xml .Description}}</description>
    <location>{{xml .Location}}</location>
  </skill>
{{- end }}
</available_skills>`))

	skillContentTemplate = template.Must(template.New("skill-content").Funcs(funcMap).Parse(`<skill_content name="{{xml .Name}}">
# Skill: {{xml .Name}}

{{.Body}}

Base directory for this skill: {{.BaseDir}}
Relative paths in this skill are relative to this base directory.
</skill_content>`))
)

func NewSkillTool(cfg SkillToolConfig) (tool.Tool, error) {
	skillFS, displayRoot, localRoot, err := resolveSkillSource(cfg)
	if err != nil {
		return nil, err
	}

	skills, err := loadSkills(skillFS, displayRoot, localRoot)
	if err != nil {
		return nil, err
	}

	byName := make(map[string]skillDefinition, len(skills))
	for _, skill := range skills {
		if existingPath, ok := byName[skill.Name]; ok {
			return nil, fmt.Errorf("duplicate skill name %q in %q and %q", skill.Name, existingPath.Location, skill.Location)
		}

		byName[skill.Name] = skill
	}

	description, err := renderCatalogDescription(skills)
	if err != nil {
		return nil, fmt.Errorf("render skill catalog description: %w", err)
	}

	return &skillTool{description: description, skills: skills, byName: byName}, nil
}

func (t *skillTool) ID() string {
	return "skill"
}

func (t *skillTool) Description() string {
	return t.description
}

func (t *skillTool) Schema() json.RawMessage {
	return skillSchema
}

func (t *skillTool) Execute(_ context.Context, args json.RawMessage) (*tool.Result, error) {
	var in skillInput
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	name := strings.TrimSpace(in.Name)
	if name == "" {
		return nil, fmt.Errorf("name is required")
	}

	skill, ok := t.byName[name]
	if !ok {
		available := make([]string, 0, len(t.skills))
		for _, s := range t.skills {
			available = append(available, s.Name)
		}

		return nil, fmt.Errorf("skill %q not found. Available skills: %s", name, strings.Join(available, ", "))
	}

	output, err := renderSkillContent(skill)
	if err != nil {
		return nil, fmt.Errorf("render skill content: %w", err)
	}

	return &tool.Result{Content: []model.ContentPart{model.NewContentText(output)}}, nil
}

func resolveSkillSource(cfg SkillToolConfig) (skillFS fs.FS, displayRoot string, localRoot string, err error) {
	if cfg.FS != nil {
		root := strings.TrimSpace(cfg.SkillsDir)
		if root == "" {
			root = "."
		}

		return cfg.FS, root, "", nil
	}

	root := strings.TrimSpace(cfg.SkillsDir)
	if root == "" {
		return nil, "", "", fmt.Errorf("skills dir is required when fs is not provided")
	}

	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, "", "", fmt.Errorf("resolve skills dir absolute path: %w", err)
	}

	info, err := os.Stat(absRoot)
	if err != nil {
		return nil, "", "", fmt.Errorf("stat skills dir: %w", err)
	}

	if !info.IsDir() {
		return nil, "", "", fmt.Errorf("skills dir %q is not a directory", root)
	}

	return os.DirFS(absRoot), filepath.ToSlash(absRoot), absRoot, nil
}

func loadSkills(skillFS fs.FS, displayRoot string, localRoot string) ([]skillDefinition, error) {
	skills := make([]skillDefinition, 0)

	err := fs.WalkDir(skillFS, ".", func(skillPath string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() || d.Name() != "SKILL.md" {
			return nil
		}

		skill, err := parseSkillFile(skillFS, skillPath, displayRoot, localRoot)
		if err != nil {
			return fmt.Errorf("loading %q: %w", skillPath, err)
		}

		skills = append(skills, skill)

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("scan skills: %w", err)
	}

	sort.Slice(skills, func(i, j int) bool { return skills[i].Name < skills[j].Name })

	return skills, nil
}

func parseSkillFile(skillFS fs.FS, skillPath string, displayRoot string, localRoot string) (skillDefinition, error) {
	b, err := fs.ReadFile(skillFS, skillPath)
	if err != nil {
		return skillDefinition{}, fmt.Errorf("read file: %w", err)
	}

	frontmatter, body, err := splitFrontmatter(string(b))
	if err != nil {
		return skillDefinition{}, err
	}

	fm, err := parseFrontmatter(frontmatter)
	if err != nil {
		return skillDefinition{}, fmt.Errorf("parse frontmatter: %w", err)
	}

	name := strings.TrimSpace(fm.Name)
	if name == "" {
		return skillDefinition{}, fmt.Errorf("frontmatter field \"name\" is required")
	}

	description := strings.TrimSpace(fm.Description)
	if description == "" {
		return skillDefinition{}, fmt.Errorf("frontmatter field \"description\" is required")
	}

	location, baseDir, err := skillLocation(skillPath, displayRoot, localRoot)
	if err != nil {
		return skillDefinition{}, err
	}

	return skillDefinition{
		Name:        name,
		Description: description,
		Body:        strings.TrimSpace(body),
		Location:    location,
		BaseDir:     baseDir,
	}, nil
}

func skillLocation(skillPath string, displayRoot string, localRoot string) (location string, baseDir string, err error) {
	if localRoot != "" {
		fullPath := filepath.Join(localRoot, filepath.FromSlash(skillPath))
		absPath, err := filepath.Abs(fullPath)
		if err != nil {
			return "", "", fmt.Errorf("resolve absolute path: %w", err)
		}

		return absPath, filepath.Dir(absPath), nil
	}

	cleanPath := path.Clean(skillPath)
	if displayRoot != "" && displayRoot != "." {
		cleanPath = path.Join(path.Clean(displayRoot), cleanPath)
	}

	return cleanPath, path.Dir(cleanPath), nil
}

func splitFrontmatter(raw string) (string, string, error) {
	raw = strings.ReplaceAll(raw, "\r\n", "\n")
	lines := strings.Split(raw, "\n")

	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return "", "", fmt.Errorf("missing frontmatter start delimiter")
	}

	end := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			end = i
			break
		}
	}

	if end == -1 {
		return "", "", fmt.Errorf("missing frontmatter end delimiter")
	}

	frontmatter := strings.Join(lines[1:end], "\n")
	body := strings.TrimLeft(strings.Join(lines[end+1:], "\n"), "\n")

	return frontmatter, body, nil
}

func parseFrontmatter(raw string) (skillFrontmatter, error) {
	result := skillFrontmatter{}

	for i, line := range strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		key, value, ok := strings.Cut(line, ":")
		if !ok {
			return skillFrontmatter{}, fmt.Errorf("invalid line %d: %q", i+1, line)
		}

		key = strings.TrimSpace(key)
		if key == "" {
			return skillFrontmatter{}, fmt.Errorf("invalid line %d: missing key", i+1)
		}

		normalizedValue := normalizeFrontmatterValue(value)

		switch key {
		case "name":
			result.Name = normalizedValue
		case "description":
			result.Description = normalizedValue
		}
	}

	return result, nil
}

func normalizeFrontmatterValue(value string) string {
	value = strings.TrimSpace(value)

	if len(value) >= 2 {
		if strings.HasPrefix(value, `"`) && strings.HasSuffix(value, `"`) {
			return strings.TrimSpace(value[1 : len(value)-1])
		}

		if strings.HasPrefix(value, `'`) && strings.HasSuffix(value, `'`) {
			return strings.TrimSpace(value[1 : len(value)-1])
		}
	}

	return value
}

func renderCatalogDescription(skills []skillDefinition) (string, error) {
	if len(skills) == 0 {
		return noSkillsDescription, nil
	}

	var b bytes.Buffer
	err := catalogTemplate.Execute(&b, struct {
		Skills []skillDefinition
	}{Skills: skills})
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(b.String()), nil
}

func renderSkillContent(skill skillDefinition) (string, error) {
	var b bytes.Buffer
	err := skillContentTemplate.Execute(&b, skill)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(b.String()), nil
}

func escapeXML(v string) string {
	return xmlEscaper.Replace(v)
}
