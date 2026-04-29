#!/bin/bash
set -e

echo "🚀 Setting up FLPG Project..."

# Create directories
echo "📁 Creating directory structure..."
mkdir -p data/validation/{main,corner,golden}
mkdir -p data/models

# Setup backend
echo "🐍 Setting up Python backend..."
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements.policy.txt
cd ..

# Setup frontend
echo "⚛️ Setting up React frontend..."
cd frontend
npm install
cd ..

# Setup contracts
echo "🔗 Setting up Smart Contracts..."
cd contracts
npm install
cd ..

# Copy environment file
echo "🔐 Creating environment file..."
cp .env.example .env

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Start the full local stack + simulated gradients: ./scripts/run_demo.sh"
echo "  3. Optional Docker stack only: docker compose up -d"
