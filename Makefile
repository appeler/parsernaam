.PHONY: help clean dev-install lint format type-check docstring-check test docs docs-serve build ci

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

dev-install: ## Install package with development dependencies
	uv sync --all-groups --all-extras

test: ## Run tests
	uv run pytest

lint: ## Run linter
	uv run ruff check .
	uv run ruff format --check .

format: ## Format code
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run type checker
	uv run pyright

docstring-check: ## Check docstring signatures
	uv run pydoclint --config=pyproject.toml src

docs: ## Build documentation
	uv run sphinx-build -W --keep-going -b html docs docs/_build/html

docs-serve: ## Serve documentation locally
	uv run python -m http.server --directory docs/_build/html 8000

build: ## Build package
	uv build

ci: lint type-check docstring-check test ## Run the local CI gate
