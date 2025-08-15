#!/bin/bash
set -e

echo "🔧 Running code formatting..."

echo "📦 Installing/syncing dependencies..."
uv sync --group dev

echo "🔧 Running isort..."
uv run isort backend/ main.py

echo "🖤 Running black..."
uv run black backend/ main.py

echo "✅ Code formatting complete!"