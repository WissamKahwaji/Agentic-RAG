.DEFAULT_GOAL := help
BACKEND := backend
FRONTEND := frontend
UV := cd $(BACKEND) && uv run

.PHONY: help setup up down reset migrate dev web ask corpus ingest lint format typecheck test test-all check web-check

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install backend and frontend dependencies, create .env if missing
	@test -f .env || cp .env.example .env
	cd $(BACKEND) && uv sync
	cd $(FRONTEND) && npm ci

up: ## Start infrastructure (Postgres + pgvector) and wait until healthy
	docker compose up -d --wait

down: ## Stop infrastructure (data is kept)
	docker compose down

reset: ## Stop infrastructure and DELETE the database volume
	docker compose down --volumes

migrate: ## Apply database migrations
	$(UV) alembic upgrade head

corpus: ## Download the public sample corpus into data/raw
	./scripts/fetch_sample_corpus.sh data/raw

ingest: ## Ingest documents: make ingest DOCS=data/raw
	$(UV) python -m agentic_rag.ingestion ../$(or $(DOCS),data/raw)

dev: ## Run the API with auto-reload on http://localhost:8000
	$(UV) uvicorn agentic_rag.main:app --reload --port 8000

web: ## Run the chat UI on http://localhost:3000 (needs `make dev` running)
	cd $(FRONTEND) && npm run dev

ask: ## Ask the running API a question: make ask Q="How do I create an HNSW index?"
	@cd $(BACKEND) && uv run python -m agentic_rag.ask_cli "$(Q)"

lint: ## Lint and check formatting
	$(UV) ruff check . && uv run ruff format --check .

format: ## Auto-fix lint issues and format
	$(UV) ruff check --fix . && uv run ruff format .

typecheck: ## Static type check (mypy, strict)
	$(UV) mypy

test: ## Unit tests (no external services)
	$(UV) pytest -m "not integration"

test-all: ## All tests, including integration (needs `make up` and Ollama)
	$(UV) pytest

web-check: ## Frontend lint, type-check and tests
	cd $(FRONTEND) && npm run lint && npm run typecheck && npm test

check: lint typecheck test web-check ## Everything CI runs
