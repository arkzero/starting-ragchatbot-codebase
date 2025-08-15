#!/bin/bash
set -e

echo "🔍 Running code quality checks..."

echo "📦 Installing/syncing dependencies..."
uv sync --group dev

echo "🔍 Running flake8..."
uv run flake8 backend/ main.py

echo "🔍 Running mypy..."
uv run mypy backend/ main.py

echo "🔍 Running isort check..."
uv run isort --check-only backend/ main.py

echo "🔍 Running black check..."
uv run black --check backend/ main.py

echo "✅ All quality checks passed!"