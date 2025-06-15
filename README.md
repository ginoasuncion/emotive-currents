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
