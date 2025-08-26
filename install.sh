#!/bin/bash

set -e  # Exit on error

echo "🚀 Installing NaviGraph with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source UV environment
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    # Check if UV is now available
    if ! command -v uv &> /dev/null; then
        echo "❌ UV installation failed. Please install UV manually:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    echo "✅ UV installed successfully!"
else
    echo "✅ UV already installed"
fi

# Install the project with uv sync
echo "📥 Installing NaviGraph with uv sync..."
uv sync

echo ""
echo "🎉 Installation complete!"
echo ""
echo "You can now use NaviGraph:"
echo "  uv run navigraph --help"
echo "  uv run navigraph setup graph config.yaml"
echo "  uv run navigraph run config.yaml"
echo ""
echo "Or activate the environment and use directly:"
echo "  source .venv/bin/activate"
echo "  navigraph --help"