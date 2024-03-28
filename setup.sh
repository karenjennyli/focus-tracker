#!/bin/bash

# Activate virtual environment
echo "Activating virtual environment..."
source "$FOCUS_VENV_PATH"

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd "$FRONTEND_DIR"
npm install

# Create database
echo "Creating database..."
cd "$BACKEND_DIR"
python3 manage.py makemigrations && python3 manage.py migrate

# After this:
# cd "$BACKEND_DIR" && python3 manage.py runserver
# cd "$FRONTEND_DIR" && npm start
# http://localhost:3000