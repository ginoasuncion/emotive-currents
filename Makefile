.PHONY: help install install-dev format lint test clean pre-commit-install

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install development dependencies
	uv sync --dev

format: ## Format code with black and ruff
	uv run ruff format .
	uv run black .

lint: ## Run linting checks
	uv run ruff check .
	uv run black --check --diff .
	uv run mypy src/

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

clean: ## Clean up cache and temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

check: lint test ## Run all checks (lint + test)

fix: ## Auto-fix code issues where possible
	uv run ruff check --fix .
	uv run ruff format .
	uv run black .
