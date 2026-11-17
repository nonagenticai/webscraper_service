#!/bin/bash
# WebScraper Service Startup Script

set -e

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9119}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
WORKERS="${WORKERS:-1}"

echo "Starting WebScraper Service..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log Level: $LOG_LEVEL"
echo "Workers: $WORKERS"

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    fi
fi

# Install playwright browsers if not present
if [ ! -d "${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}" ]; then
    echo "Installing Playwright browsers..."
    playwright install chromium
fi

# Run the application
exec uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "${LOG_LEVEL,,}" \
    --workers "$WORKERS"
