#!/bin/bash
# Startup script for Local RAG Agent
# Ensures proper Python path setup

echo "Starting Local RAG Agent..."

# Change to script directory
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated."
fi

# Check Ollama
echo "Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "WARNING: Ollama does not appear to be running!"
    echo "Please start Ollama first: ollama serve"
    read -p "Press Enter to continue anyway..."
fi

# Start Streamlit
echo "Launching Streamlit UI..."
streamlit run app.py
