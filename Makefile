# Wine Quality ML Platform — developer tasks.
#
# `make help` lists everything available.

.DEFAULT_GOAL := help
.PHONY: help install train test test-fast lint format check clean docker-build docker-up docker-train docker-down docker-logs api dashboard

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

install: ## Create the virtualenv and install all dependencies
	uv sync --extra dev

train: ## Train both models and write artifacts to models/
	$(PYTHON) -m wine_quality.models.train

test: ## Run the full test suite with coverage
	$(PYTEST)

test-fast: ## Run tests, skipping the slow end-to-end training tests
	$(PYTEST) -m "not slow"

lint: ## Check code style and imports
	$(RUFF) check .

format: ## Auto-fix style issues and format code
	$(RUFF) check --fix .
	$(RUFF) format .

check: lint test ## Run lint and the full test suite

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f models/*.joblib models/*.json reports/*.json reports/*.jsonl

# ---------------------------------------------------------------------------
# Local services
# ---------------------------------------------------------------------------

api: ## Run the prediction API locally on port 8000
	uv run uvicorn wine_quality.serving.api:app --reload --port 8000

dashboard: ## Run the Streamlit dashboard locally on port 8501
	uv run streamlit run app/Overview.py --server.port 8501

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the images
	docker compose build

docker-train: ## Run training inside Docker (writes to the shared volume)
	docker compose --profile train run --rm train

docker-up: ## Start the API and dashboard
	docker compose up -d

docker-down: ## Stop everything and remove containers
	docker compose down

docker-logs: ## Tail logs from all services
	docker compose logs -f