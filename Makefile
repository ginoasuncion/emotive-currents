.PHONY: help install install-dev format lint test clean env activate shell

# Colors for output
GREEN=\033[0;32m
YELLOW=\033[1;33m
NC=\033[0m # No Color

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment Management
env: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	uv sync --dev
	@echo "$(GREEN)✅ Environment created and dependencies installed$(NC)"
	@echo "$(YELLOW)To activate: source .venv/bin/activate$(NC)"

activate: ## Show activation command
	@echo "$(YELLOW)To activate the environment, run:$(NC)"
	@echo "source .venv/bin/activate"
	@echo ""
	@echo "$(YELLOW)Or on Windows:$(NC)"
	@echo ".venv\\Scripts\\activate"
	@echo ""
	@echo "$(YELLOW)To deactivate:$(NC)"
	@echo "deactivate"

shell: ## Activate environment and start shell
	@echo "$(GREEN)Starting shell with activated environment...$(NC)"
	@bash --rcfile <(echo "source .venv/bin/activate; echo 'Environment activated! Type exit to return.'")

env-info: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python version: $$(uv run python --version)"
	@echo "Virtual env path: $$(pwd)/.venv"
	@echo "Activated: $$(if [ "$$VIRTUAL_ENV" ]; then echo "Yes ($$VIRTUAL_ENV)"; else echo "No"; fi)"
	@echo "Installed packages:"
	@uv pip list | head -10

# Installation
install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install development dependencies
	uv sync --dev

# Code Quality
format: ## Format code with black and ruff
	uv run ruff format .
	uv run black .

lint: ## Run linting checks
	uv run ruff check .
	uv run black --check --diff .
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html --cov-report=term

# Utilities
clean: ## Clean up cache and temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage coverage.xml

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

check: lint test ## Run all checks

dev-setup: install-dev pre-commit-install ## Complete development setup
	@echo "$(GREEN)✅ Development setup complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. make activate  # See activation instructions"
	@echo "  2. make test      # Run tests"
	@echo "  3. make format    # Format code"

# Manual environment activation (requires sourcing)
.venv/bin/activate: pyproject.toml
	uv sync --dev
	touch .venv/bin/activate

# Development workflow
dev: .venv/bin/activate ## Set up development environment
	@echo "$(GREEN)Development environment ready!$(NC)"
	@$(MAKE) activate
