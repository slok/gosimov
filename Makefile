SHELL := $(shell which bash)

UNIT_TEST_CMD := ./scripts/check/unit-test.sh
CHECK_CMD := ./scripts/check/check.sh

help: ## Show this help
	@echo "Help"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[93m %s\n", $$1, $$2}'

.PHONY: default
default: help

.PHONY: test
test: ## Runs unit tests.
	@$(UNIT_TEST_CMD)

.PHONY: check
check: ## Runs checks (linting).
	@$(CHECK_CMD)

.PHONY: fmt
fmt: ## Formats code.
	gofmt -s -w .
	goimports -w .

.PHONY: go-gen
go-gen: ## Generates Go code (mocks, etc.).
	mockery
	go generate ./...

.PHONY: ci
ci: check test ## Runs all CI checks.
