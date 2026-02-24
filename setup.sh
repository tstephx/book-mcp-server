#!/bin/bash

# Production-ready setup script for Book MCP Server

echo "📚 Setting up Book Library MCP Server (Production-Ready)"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Test the server: python server.py"
echo "2. Configure Claude Desktop (see docs/QUICKSTART.md)"
echo "3. Restart Claude Desktop"
echo ""
