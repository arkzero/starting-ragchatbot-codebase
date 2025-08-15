#!/bin/bash
set -e

echo "🚀 Running full quality workflow..."

echo "📦 Installing/syncing dependencies..."
uv sync --group dev

echo "🔧 Step 1: Formatting code..."
uv run isort backend/ main.py
uv run black backend/ main.py

echo "🔍 Step 2: Running quality checks..."
uv run flake8 backend/ main.py
uv run mypy backend/ main.py

echo "🧪 Step 3: Running tests..."
uv run pytest

echo "✅ Full quality workflow complete!"