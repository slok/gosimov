---
name: git-release
description: Prepare a release summary and suggest release commands from repository history.
---

## When to use

Use this skill when the user asks for release notes, changelog drafts, or version bump proposals.

## Workflow

1. Inspect repository status and recent commits with shell commands.
2. Check release-related files such as `README.md`, `CHANGELOG.md`, and `.github` workflows.
3. Propose a semantic version bump with rationale.
4. Draft concise markdown release notes with highlights and breaking changes.

## Output checklist

- Proposed version and reason
- Key highlights
- Breaking changes (if any)
- Suggested `gh release create` command
