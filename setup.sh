#!/bin/bash

# Emotive Currents Project Setup Script
# This script creates a clean project structure

echo "🚀 Setting up Emotive Currents project..."

# Remove existing files that might cause conflicts
echo "📁 Cleaning up existing files..."
rm -f requirements.txt
rm -f pyproject.toml
rm -rf .venv
rm -rf src/__pycache__
rm -rf **/__pycache__

# Create directory structure
echo "📂 Creating directory structure..."
mkdir -p src/emotive_currents
mkdir -p tests
mkdir -p .github/workflows
mkdir -p notebooks
mkdir -p scripts
mkdir -p api
mkdir -p data
mkdir -p docs

# Create package files
echo "📝 Creating package files..."

# Main package __init__.py
cat > src/emotive_currents/__init__.py << 'EOF'
"""Emotive Currents - A project for analyzing emotive currents."""

__version__ = "0.1.0"
__author__ = "Gino Asuncion"
__email__ = "gino@example.com"

# Main imports will go here as you develop the package
# from .core import *
# from .utils import *
EOF

# Create a sample core module
cat > src/emotive_currents/core.py << 'EOF'
"""Core functionality for emotive currents analysis."""


def hello_world() -> str:
    """Sample function to test the package setup."""
    return "Hello from Emotive Currents!"


def analyze_emotion(text: str) -> dict:
    """
    Placeholder function for emotion analysis.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with emotion analysis results
    """
    return {
        "text": text,
        "emotions": {"positive": 0.5, "negative": 0.3, "neutral": 0.2},
        "confidence": 0.8
    }
EOF

# Create utils module
cat > src/emotive_currents/utils.py << 'EOF'
"""Utility functions for the emotive currents package."""

import logging
from typing import Any, Dict, List


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def validate_input(data: Any) -> bool:
    """Validate input data."""
    if data is None:
        return False
    if isinstance(data, str) and len(data.strip()) == 0:
        return False
    return True


def process_batch(items: List[str]) -> List[Dict]:
    """Process a batch of items."""
    results = []
    for item in items:
        if validate_input(item):
            results.append({"item": item, "processed": True})
        else:
            results.append({"item": item, "processed": False, "error": "Invalid input"})
    return results
EOF

# Create sample test files
echo "🧪 Creating test files..."

cat > tests/__init__.py << 'EOF'
"""Tests for emotive currents package."""
EOF

cat > tests/test_core.py << 'EOF'
"""Tests for core functionality."""

from emotive_currents.core import hello_world, analyze_emotion


def test_hello_world():
    """Test the hello world function."""
    result = hello_world()
    assert result == "Hello from Emotive Currents!"


def test_analyze_emotion():
    """Test emotion analysis function."""
    result = analyze_emotion("I am happy today!")
    
    assert "text" in result
    assert "emotions" in result
    assert "confidence" in result
    assert result["text"] == "I am happy today!"
    assert isinstance(result["emotions"], dict)
    assert isinstance(result["confidence"], (int, float))


def test_analyze_emotion_empty():
    """Test emotion analysis with empty text."""
    result = analyze_emotion("")
    
    assert result["text"] == ""
    assert "emotions" in result
EOF

cat > tests/test_utils.py << 'EOF'
"""Tests for utility functions."""

from emotive_currents.utils import validate_input, process_batch


def test_validate_input_valid():
    """Test validation with valid input."""
    assert validate_input("valid text") is True
    assert validate_input(123) is True
    assert validate_input([1, 2, 3]) is True


def test_validate_input_invalid():
    """Test validation with invalid input."""
    assert validate_input(None) is False
    assert validate_input("") is False
    assert validate_input("   ") is False


def test_process_batch():
    """Test batch processing."""
    items = ["valid", "", "also valid", None]
    results = process_batch(items)
    
    assert len(results) == 4
    assert results[0]["processed"] is True
    assert results[1]["processed"] is False
    assert results[2]["processed"] is True
    assert results[3]["processed"] is False
EOF

# Create sample notebook
echo "📓 Creating sample notebook structure..."
cat > notebooks/README.md << 'EOF'
# Notebooks

This directory contains Jupyter notebooks for data exploration and analysis.

## Structure

- `exploratory/` - Initial data exploration notebooks
- `analysis/` - Main analysis notebooks  
- `experiments/` - Experimental work and prototyping
- `reports/` - Final analysis and reporting notebooks
EOF

mkdir -p notebooks/exploratory
mkdir -p notebooks/analysis
mkdir -p notebooks/experiments
mkdir -p notebooks/reports

# Create sample script
echo "📜 Creating sample scripts..."
cat > scripts/README.md << 'EOF'
# Scripts

This directory contains utility scripts for the project.

- `data_processing/` - Data preprocessing and cleaning scripts
- `training/` - Model training scripts
- `evaluation/` - Model evaluation scripts
- `deployment/` - Deployment and serving scripts
EOF

mkdir -p scripts/data_processing
mkdir -p scripts/training
mkdir -p scripts/evaluation
mkdir -p scripts/deployment

# Create API structure
echo "🌐 Creating API structure..."
cat > api/README.md << 'EOF'
# API

This directory contains API-related code.

- `main.py` - Main API application
- `routes/` - API route definitions
- `models/` - Pydantic models for request/response
- `services/` - Business logic services
EOF

mkdir -p api/routes
mkdir -p api/models
mkdir -p api/services

# Create updated README
echo "📖 Creating README..."
cat > README.md << 'EOF'
# Emotive Currents

A project for analyzing emotive currents in text data.

## Setup

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup the project**:
   ```bash
   git clone https://github.com/ginoasuncion/emotive-currents.git
   cd emotive-currents
   uv sync --dev
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   uv run pre-commit install
   ```

## Development

### Running tests
```bash
uv run pytest
```

### Code formatting
```bash
uv run black .
uv run ruff format .
```

### Linting
```bash
uv run ruff check .
```

### All checks
```bash
make check  # Runs linting and tests
```

## Project Structure

```
emotive-currents/
├── src/emotive_currents/    # Main package
├── tests/                   # Test files
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── api/                    # API code
├── data/                   # Data files (gitignored)
├── docs/                   # Documentation
└── .github/workflows/      # CI/CD workflows
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. cd emotive-currents"
echo "2. uv sync --dev"
echo "3. uv run pytest  # Run tests"
echo "4. uv run pre-commit install  # Setup git hooks"
echo ""
echo "Your project is ready! 🎉"
