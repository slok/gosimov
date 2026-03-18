package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/slok/gosimov/pkg/model"
	"github.com/slok/gosimov/pkg/tool"
	toolschema "github.com/slok/gosimov/pkg/tool/schema"
)

type SkillToolConfig struct {
	SkillsDir string
	Debug     bool
}

type skillTool struct {
	skills []skillDefinition
	byName map[string]skillDefinition
	debug  bool
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
)

func NewSkillTool(cfg SkillToolConfig) (tool.Tool, error) {
	root := strings.TrimSpace(cfg.SkillsDir)
	if root == "" {
		return nil, fmt.Errorf("skills dir is required")
	}

	debugf(cfg.Debug, "initializing skill tool (skills_dir=%s)", root)

	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("stat skills dir: %w", err)
	}

	if !info.IsDir() {
		return nil, fmt.Errorf("skills dir %q is not a directory", root)
	}

	skills, err := loadSkills(root)
	if err != nil {
		return nil, err
	}

	byName := make(map[string]skillDefinition, len(skills))
	for _, skill := range skills {
		if existingPath, ok := byName[skill.Name]; ok {
			return nil, fmt.Errorf("duplicate skill name %q in %q and %q", skill.Name, existingPath.Location, skill.Location)
		}

		byName[skill.Name] = skill
		debugf(cfg.Debug, "registered skill (name=%s location=%s)", skill.Name, skill.Location)
	}

	debugf(cfg.Debug, "skill tool initialized (skills=%d)", len(skills))

	return &skillTool{skills: skills, byName: byName, debug: cfg.Debug}, nil
}

func (t *skillTool) ID() string {
	return "skill"
}

func (t *skillTool) Description() string {
	debugf(t.debug, "building skill catalog for description (skills=%d)", len(t.skills))

	if len(t.skills) == 0 {
		return "Load a specialized skill that provides domain-specific instructions and workflows. No skills are currently available."
	}

	lines := []string{
		"Load a specialized skill that provides domain-specific instructions and workflows.",
		"",
		"When you recognize that a task matches one of the skills listed below, call this tool with that skill name.",
		"",
		"<available_skills>",
	}

	for _, skill := range t.skills {
		lines = append(lines,
			"  <skill>",
			fmt.Sprintf("    <name>%s</name>", escapeXML(skill.Name)),
			fmt.Sprintf("    <description>%s</description>", escapeXML(skill.Description)),
			fmt.Sprintf("    <location>%s</location>", escapeXML(skill.Location)),
			"  </skill>",
		)
	}

	lines = append(lines, "</available_skills>")

	return strings.Join(lines, "\n")
}

func (t *skillTool) Schema() json.RawMessage {
	return skillSchema
}

func (t *skillTool) Execute(_ context.Context, args json.RawMessage) (*tool.Result, error) {
	var in skillInput
	if err := toolschema.DecodeStrict(args, &in); err != nil {
		debugf(t.debug, "skill execute failed decoding args: %v", err)
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	name := strings.TrimSpace(in.Name)
	debugf(t.debug, "skill execute requested (name=%s)", name)
	if name == "" {
		debugf(t.debug, "skill execute rejected: empty name")
		return nil, fmt.Errorf("name is required")
	}

	skill, ok := t.byName[name]
	if !ok {
		available := make([]string, 0, len(t.skills))
		for _, s := range t.skills {
			available = append(available, s.Name)
		}
		debugf(t.debug, "skill execute failed: skill not found (name=%s available=%s)", name, strings.Join(available, ","))

		return nil, fmt.Errorf("skill %q not found. Available skills: %s", name, strings.Join(available, ", "))
	}

	output := strings.Join([]string{
		fmt.Sprintf(`<skill_content name="%s">`, escapeXML(skill.Name)),
		fmt.Sprintf("# Skill: %s", skill.Name),
		"",
		strings.TrimSpace(skill.Body),
		"",
		fmt.Sprintf("Base directory for this skill: %s", skill.BaseDir),
		"Relative paths in this skill are relative to this base directory.",
		"</skill_content>",
	}, "\n")

	debugf(t.debug, "skill execute loaded (name=%s location=%s)", skill.Name, skill.Location)

	return &tool.Result{Content: []model.ContentPart{model.NewContentText(output)}}, nil
}

func loadSkills(root string) ([]skillDefinition, error) {
	skills := make([]skillDefinition, 0)

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() || d.Name() != "SKILL.md" {
			return nil
		}

		skill, err := parseSkillFile(path)
		if err != nil {
			return fmt.Errorf("loading %q: %w", path, err)
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

func parseSkillFile(path string) (skillDefinition, error) {
	b, err := os.ReadFile(path)
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

	absPath, err := filepath.Abs(path)
	if err != nil {
		return skillDefinition{}, fmt.Errorf("resolve absolute path: %w", err)
	}

	return skillDefinition{
		Name:        name,
		Description: description,
		Body:        body,
		Location:    absPath,
		BaseDir:     filepath.Dir(absPath),
	}, nil
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

	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			return skillFrontmatter{}, fmt.Errorf("invalid line %d: %q", i+1, line)
		}

		key := strings.TrimSpace(parts[0])
		if key == "" {
			return skillFrontmatter{}, fmt.Errorf("invalid line %d: missing key", i+1)
		}

		value := normalizeFrontmatterValue(parts[1])

		switch key {
		case "name":
			result.Name = value
		case "description":
			result.Description = value
		}
	}

	return result, nil
}

func normalizeFrontmatterValue(v string) string {
	v = strings.TrimSpace(v)
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			v = v[1 : len(v)-1]
		}
	}

	return strings.TrimSpace(v)
}

func escapeXML(v string) string {
	return xmlEscaper.Replace(v)
}

func debugf(enabled bool, format string, args ...any) {
	if !enabled {
		return
	}

	log.Printf("[skills-tool] "+format, args...)
}
