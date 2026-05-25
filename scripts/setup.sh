#!/bin/bash
set -e

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if command_exists python3; then
    PYTHON_BOOTSTRAP=python3
elif command_exists python; then
    PYTHON_BOOTSTRAP=python
else
    echo "Python 3 is required but was not found."
    exit 1
fi

echo "🚀 Setting up FLPG Project..."

# Create directories
echo "📁 Creating directory structure..."
mkdir -p data/validation/{main,corner,golden}
mkdir -p data/models

# Setup backend
echo "🐍 Setting up Python backend..."
cd backend
"$PYTHON_BOOTSTRAP" -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements.policy.txt
cd ..

# Setup frontend
echo "⚛️ Setting up React frontend..."
cd frontend
npm ci
cd ..

# Setup contracts
echo "🔗 Setting up Smart Contracts..."
cd contracts
npm ci
cd ..

# Copy environment file
if [[ -f .env ]]; then
    echo "🔐 Keeping existing .env"
else
    echo "🔐 Creating environment file..."
    cp .env.example .env
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Start the full local stack + simulated gradients: ./scripts/run_demo.sh"
echo "  3. Optional Docker stack only: docker compose up -d"
