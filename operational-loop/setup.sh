#!/usr/bin/env bash
# One-time setup: a venv with both upstream use cases installed editable.
# The loop has no dependencies of its own — the two use cases ARE the deps.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/opt/homebrew/bin/python3.12}"

"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
# [dev] on the first pulls in pytest + ruff for tests/test_loop.py and linting.
.venv/bin/pip install --quiet -e "../delivery-commit-prediction[dev]" -e ../intervention-optimization

echo "Done. Run the loop with: .venv/bin/python run_loop.py"
