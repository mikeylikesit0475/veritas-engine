# Veritas Engine — make targets. Thin wrappers around uv; no hidden logic.

.PHONY: sync lint fmt typecheck test test-strict clean

# uv-managed venv. .venv is created in the project root by default.
UV ?= uv

sync: ## Install/refresh dev deps via uv
	$(UV) sync

lint: ## ruff check (gate)
	$(UV) run ruff check veritas tests

fmt: ## ruff format (in place)
	$(UV) run ruff format veritas tests

typecheck: ## mypy --strict (gate)
	$(UV) run mypy veritas

test: ## pytest — no live model calls; CI replays fixtures
	$(UV) run pytest

test-strict: lint typecheck test ## lint + typecheck + test — the full gate

clean: ## Remove build/test caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +