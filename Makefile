.PHONY: test test-unit test-integration test-e2e test-cov test-fast test-watch clean-test help

VENV := .venv/bin/python
PYTEST := $(VENV) -m pytest

# ─── Primary targets ────────────────────────────────────────────────────────

help:
	@echo "Test targets:"
	@echo "  make test           - Run all unit tests (default)"
	@echo "  make test-unit      - Unit tests only"
	@echo "  make test-integration - Integration tests (requires OPENAI_API_KEY + real DB)"
	@echo "  make test-e2e       - End-to-end CLI tests"
	@echo "  make test-cov       - Unit tests with HTML coverage report"
	@echo "  make test-fast      - Parallel unit tests (pytest-xdist)"
	@echo "  make test-all       - All tests including integration"
	@echo ""
	@echo "Pipeline targets:"
	@echo "  make health         - Check pipeline health"
	@echo "  make pending        - Show pending approvals"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean-test     - Remove test artifacts"

# Run unit tests (skips integration by default via pyproject.toml addopts)
test:
	$(PYTEST) tests/ -v

# Explicit unit-only run
test-unit:
	$(PYTEST) tests/ -v -m "unit or (not integration and not e2e and not slow)"

# Integration tests — requires OPENAI_API_KEY and real library DB
test-integration:
	$(PYTEST) tests/ -v -m integration

# End-to-end CLI tests
test-e2e:
	$(PYTEST) tests/ -v -m e2e

# With coverage report
test-cov:
	$(PYTEST) tests/ --cov --cov-report=term-missing --cov-report=html
	@echo "\nHTML report: htmlcov/index.html"

# Parallel execution (4 workers)
test-fast:
	$(PYTEST) tests/ -n 4

# Run everything
test-all:
	$(PYTEST) tests/ -v -m "not slow"

# Run a specific test file or pattern — usage: make test-file FILE=tests/test_approval_actions.py
test-file:
	$(PYTEST) $(FILE) -v

# ─── Pipeline shortcuts ──────────────────────────────────────────────────────

health:
	agentic-pipeline health

pending:
	agentic-pipeline pending

# ─── Cleanup ────────────────────────────────────────────────────────────────

clean-test:
	rm -rf htmlcov/ .coverage .pytest_cache/ tests/__pycache__/
	find . -name "*.pyc" -delete
