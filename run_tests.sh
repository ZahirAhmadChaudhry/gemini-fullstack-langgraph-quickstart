#!/bin/bash

echo "🎯 French Sustainability Transcript Analyzer - Setup & Test"
echo "============================================================"

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    echo "❌ Please run this script from the workspace root directory"
    echo "   (where backend folder is located)"
    exit 1
fi

# Navigate to backend
cd backend

echo "📦 Installing dependencies with uv..."
uv sync

echo ""
echo "🧪 Running refactor tests..."
uv run python test_refactor.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To test with actual data:"
echo "1. Set your Gemini API key:"
echo "   export GEMINI_API_KEY='your-api-key-here'"
echo ""
echo "2. Run the full backend test:"
echo "   cd backend"
echo "   uv run python ../test_backend.py"
echo ""
echo "3. Or start the development server:"
echo "   uv run langgraph dev"
