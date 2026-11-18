#!/bin/bash
# ─────────────────────────────────────────────────────────────
# QuantumEmbeddings API — Startup Script
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════"
echo "  QuantumEmbeddings API"
echo "  QuantumHash Corporation"
echo "═══════════════════════════════════════════"

# .env load karo agar exist karta hai
if [ -f .env ]; then
    echo "📄 Loading .env file..."
    export $(grep -v '^#' .env | grep -v '^$' | grep '=.' | xargs)
else
    echo "⚠️  .env file nahi mili. .env.example ko copy karo:"
    echo "    cp .env.example .env"
    exit 1
fi

# Python version check
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Python: $(python3 --version)"

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "💻 No GPU found — CPU mode chalega"
fi

echo ""
echo "🚀 Starting server on ${HOST:-0.0.0.0}:${PORT:-8100}..."
echo ""

# Server start
exec uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8100}" \
    --workers "${WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}" \
    --reload  # Development ke liye. Production me hata do!

# Production ke liye:
# exec uvicorn app.main:app \
#     --host "${HOST:-0.0.0.0}" \
#     --port "${PORT:-8100}" \
#     --workers 1 \
#     --log-level info