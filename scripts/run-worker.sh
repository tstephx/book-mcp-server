#!/bin/bash
# Wrapper script for launchd — sources environment before running worker

# Source API keys
if [ -f "$HOME/.env" ]; then
    source "$HOME/.env"
fi

export WATCH_DIR="$HOME/Documents/_ebooks/agentic-book-pipeline"
export PROCESSED_DIR="$HOME/Documents/_ebooks/agentic-book-pipeline/processed"

exec /Users/taylorstephens/_Projects/book-mcp-server/.venv/bin/agentic-pipeline worker
