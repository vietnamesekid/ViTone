.DEFAULT_GOAL := help
.PHONY: help install install-all format lint test check ingest clean

CONFIG   ?= configs/data.yaml
WORK_DIR ?= corpus/processed
SOURCE   ?=
WORKERS  ?=

DATA_ARGS := -c $(CONFIG) --work-dir $(WORK_DIR) $(if $(SOURCE),-s $(SOURCE)) $(if $(WORKERS),--workers $(WORKERS))

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@printf "\n  Variables: CONFIG=$(CONFIG) WORK_DIR=$(WORK_DIR) SOURCE=<name> WORKERS=<n>\n"

install: ## Install core dependencies (ingest)
	uv sync

install-all: ## Install with model-stage dependencies (torch)
	uv sync --all-extras

format: ## Format code
	uv run ruff format vitone tests
	uv run ruff check --fix vitone tests

lint: ## Lint and check formatting
	uv run ruff check vitone tests
	uv run ruff format --check vitone tests

test: ## Run tests
	uv run pytest

check: lint test ## Lint + test

ingest: ## Ingest raw media (SOURCE=<name> for one source)
	uv run vitone data ingest $(DATA_ARGS)

clean: ## Delete processed outputs (raw data is kept)
	rm -rf $(WORK_DIR)
