#!/bin/bash

echo "Starting Officer Roster Optimization API..."
echo "=========================================="

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python -m venv venv
fi

source venv/bin/activate

if [ ! -f ".env" ]; then
    echo "Environment file not found. Copying from .env.example..."
    cp .env.example .env
fi

pip install -q -r requirements.txt

echo ""
echo "Starting server on http://localhost:8000"
echo "API Documentation: http://localhost:8000/api/v1/docs"
echo "=========================================="
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000