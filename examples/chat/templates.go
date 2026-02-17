package main

import (
	"embed"
	"html/template"
	"time"
)

//go:embed templates/*.html
var templatesFS embed.FS

var (
	htmlTemplates = template.Must(template.New("chat-templates").Funcs(template.FuncMap{
		"fmtTime": func(t time.Time) string {
			if t.IsZero() {
				return ""
			}

			return t.Format("2006-01-02 15:04:05")
		},
	}).ParseFS(templatesFS, "templates/*.html"))
)
