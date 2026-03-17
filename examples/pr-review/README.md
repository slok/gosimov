# PR Review Example

This example runs an automated PR review flow with gosimov and GitHub CLI.

- Trigger mention: `@gosimov-review`
- Trusted trigger authors only: `OWNER`, `MEMBER`, `COLLABORATOR`
- No generic shell tool exposed to the model
- The model can read PR context and publish review comments through dedicated GH tools

## Run locally

```bash
go run ./examples/pr-review \
  --api-key "$OPENCODE_GO_API_KEY" \
  --repo owner/repo \
  --pr 123
```

Optional dry-run mode:

```bash
go run ./examples/pr-review \
  --api-key "$OPENCODE_GO_API_KEY" \
  --repo owner/repo \
  --pr 123 \
  --dry-run
```

## GitHub Actions

Workflow: `.github/workflows/pr-review.yml`

- `pull_request_target` on PR title/body mention
- `issue_comment` on PR comments mention
- Uses `GH_TOKEN` from `secrets.GITHUB_TOKEN`
- Uses OpenCode Go key from `secrets.OPENCODE_GO_API_KEY`
