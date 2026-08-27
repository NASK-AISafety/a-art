# A-ART Framework Makefile

.PHONY: help install lock sync format lint typecheck test quality verify-setup clean \
        check-vllm run-attack run-attack-no-guard run-attack-vlm slurm-script all

# Default target
help:
	@echo "A-ART Framework - Available Commands"
	@echo "====================================="
	@echo ""
	@echo "Setup & Dependencies:"
	@echo "  make install             - Full install (lock + sync + editable pip install)"
	@echo "  make lock                - Lock dependencies (uv lock)"
	@echo "  make sync                - Sync dependencies (uv sync)"
	@echo ""
	@echo "Quality Gate:"
	@echo "  make format              - Format code (ruff format)"
	@echo "  make lint                - Lint and fix (ruff check --fix)"
	@echo "  make typecheck           - Type check (mypy --strict)"
	@echo "  make test                - Run tests (pytest)"
	@echo "  make quality             - format + lint"
	@echo ""
	@echo "Verification:"
	@echo "  make verify-setup        - Import smoke test (scripts/test_setup.py)"
	@echo "  make run-attack          - Debug run, 3 attacks, with input guard"
	@echo "  make run-attack-no-guard - Debug run, 3 attacks, no input guard (faster)"
	@echo "  make run-attack-vlm      - VLM smoke test (debug_vlm.yaml, 2 attacks)"
	@echo "  make slurm-script        - Print an example HPC Slurm script to stdout"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean               - Remove caches, build artefacts, debug JSONL"
	@echo "  make all                 - quality + verify-setup"

# =============================================================================
# STEP A: Dependency Management
# =============================================================================

install: lock sync
	@echo "Installing package in editable mode..."
	uv pip install -e .
	@echo "✓ Dependencies installed"

lock:
	@echo "Locking dependencies..."
	uv lock

sync:
	@echo "Syncing dependencies..."
	uv sync

# =============================================================================
# STEP B: Quality Gate
# =============================================================================

format:
	@echo "Formatting code..."
	uv run ruff format .

lint:
	@echo "Linting code..."
	uv run ruff check . --fix

typecheck:
	@echo "Type checking..."
	uv run mypy src/llm_red_team/ --ignore-missing-imports

test:
	@echo "Running tests..."
	uv run pytest tests/ -v --tb=short

quality: format lint
	@echo ""
	@echo "✓ Quality gate passed (format + lint)"
	@echo "  Run 'make typecheck' for strict type checking"
	@echo "  Run 'make test' for unit tests"

# =============================================================================
# STEP C: Reality Check (E2E Verification)
# =============================================================================

verify-setup:
	@echo "Verifying framework setup (import smoke test)..."
	uv run python scripts/test_setup.py

# Preflight: confirm a vLLM server is up before launching attacks.
# Uses VLLM_BASE_URL if set, otherwise tries http://localhost:8000.
VLLM_URL ?= $(or $(VLLM_BASE_URL),http://localhost:8000)

check-vllm:
	@echo "Checking vLLM server at $(VLLM_URL)..."
	@curl -sf --max-time 3 "$(VLLM_URL)/v1/models" > /dev/null 2>&1 || \
		( echo ""; \
		  echo "  ERROR: No vLLM server reachable at $(VLLM_URL)"; \
		  echo ""; \
		  echo "  Start a local server first:"; \
		  echo "    bash scripts/start_local_vllm.sh"; \
		  echo ""; \
		  echo "  Or point to an existing server:"; \
		  echo "    export VLLM_BASE_URL=http://<host>:8000"; \
		  echo "    make run-attack"; \
		  echo ""; \
		  exit 1 )
	@echo "  ✓ vLLM server is reachable — served models:"
	@curl -sf --max-time 3 "$(VLLM_URL)/v1/models" | \
		python3 -c "import sys,json; [print('      -', m['id']) for m in json.load(sys.stdin).get('data', [])]" 2>/dev/null || true

# Standard debug run — requires VLLM_BASE_URL, input guard enabled
run-attack: check-vllm
	@echo "Running attack generation (debug mode, with input guard)..."
	uv run python scripts/main.py \
		--config configs/attack_generation/debug.yaml \
		--num-attacks 3

# Faster debug run without input guard (skips PromptGuard-2 load)
run-attack-no-guard: check-vllm
	@echo "Running attack generation (debug mode, no input guard)..."
	uv run python scripts/main.py \
		--config configs/attack_generation/debug.yaml \
		--no-input-guard \
		--num-attacks 3

# VLM smoke test — requires configs/attack_generation/debug_vlm.yaml
run-attack-vlm: check-vllm
	@echo "Running VLM attack generation (debug_vlm.yaml, 2 attacks)..."
	uv run python scripts/main.py \
		--config configs/attack_generation/debug_vlm.yaml \
		--no-input-guard \
		--num-attacks 2

# Print an example Slurm script to stdout (no pipeline execution)
slurm-script:
	@echo "Generating example Slurm script (stdout)..."
	uv run python scripts/main.py \
		--config configs/attack_generation/hpc_production.yaml \
		--generate-slurm - \
		--slurm-partition gpu \
		--slurm-account my_account \
		--slurm-time 08:00:00 \
		--slurm-mem 480GB \
		--vllm-base-url http://node01:8000 \
		--num-attacks 5000

# =============================================================================
# Utilities
# =============================================================================

clean:
	@echo "Cleaning generated files..."
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov build dist *.egg-info
	rm -f outputs/agentic_debug/*.jsonl
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

# =============================================================================
# Full Pipeline
# =============================================================================

all: quality verify-setup
	@echo ""
	@echo "====================================="
	@echo "✓ All checks passed!"
	@echo "====================================="
