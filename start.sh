#!/bin/bash

# Navigate to script directory (if script is double-clicked from elsewhere)
cd "$(dirname "$0")"

# Activate virtual environment
source ./venv/bin/activate

# Run Python script
python run_gradio_offline.py
