# Veritas Engine — make targets. Thin wrappers around uv; no hidden logic.

.PHONY: sync lint fmt typecheck test test-strict record eval calibration clean

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

record: ## Record an eval fixture: make record CLAIM="..." NAME="fixture_name"
	@test -n "$(CLAIM)" || (echo "usage: make record CLAIM=\"...\" NAME=\"name\""; exit 2)
	@test -n "$(NAME)" || (echo "usage: make record CLAIM=\"...\" NAME=\"name\""; exit 2)
	VERITAS_RECORD=1 $(UV) run python -m veritas.eval_harness.record "$(CLAIM)" --name $(NAME)

eval: ## Run the eval (live; requires Ollama or Gemini) — prints the table
	$(UV) run python -m evals.run_eval

calibration: ## Render the calibration plot from the last eval run
	$(UV) run python -m evals.calibration_plot

clean: ## Remove build/test caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +