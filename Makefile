# Common developer tasks. Run `make help` for the list.
# These mirror what CI does (.github/workflows/ci.yml) so `make check` and
# `make notebook` reproduce the two CI jobs locally.

.DEFAULT_GOAL := help

.PHONY: help sync test lint format format-check notebook check ui clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## Install dependencies (incl. dev group) into the uv environment
	uv sync --dev

test: ## Run the test suite
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format the code with ruff
	uv run ruff format .

format-check: ## Check formatting without changing files
	uv run ruff format --check .

check: lint format-check test ## Lint, check formatting, and test (the lint-and-test CI job)

notebook: ## Re-execute pyfixest_regression_example.ipynb and strip volatile metadata (the notebook CI job)
	uv run jupyter execute pyfixest_regression_example.ipynb --output=pyfixest_regression_example.ipynb
	uv run nbstripout --keep-output pyfixest_regression_example.ipynb

ui: ## Open the MLflow UI on the local mlflow.db
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

clean: ## Remove the local MLflow store and caches
	rm -rf mlflow.db mlruns .pytest_cache .ruff_cache
