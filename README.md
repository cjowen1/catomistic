# Catomistic

A package for atomistic simulations and machine learning in catalysis and surface chemistry.

## Project Structure

```
catomistic/
├── src/
│   └── catomistic/  # Main package code
│       ├── core/    # Core functionality
│       ├── models/  # ML models
│       └── utils/   # Utility functions
├── tests/           # Test files
├── examples/        # Example notebooks and scripts
├── docs/           # Documentation
├── pyproject.toml  # Package configuration
└── README.md
```

## Installation

### Basic Installation

```bash
pip install .
```

### Development Installation

For development, install with additional tools:

```bash
pip install -e ".[dev]"
```

### With Visualization Support

To install with visualization support:

```bash
pip install -e ".[viz]"
```

### Documentation Development

To install with documentation tools:

```bash
pip install -e ".[docs]"
```

### Full Installation (All Features)

To install with all optional dependencies:

```bash
pip install -e ".[dev,viz,docs]"
```

## Requirements

- Python >=3.9
- PyTorch >=2.0.0
- Other dependencies are handled automatically by pip

## Development

1. Clone the repository:
```bash
git clone https://github.com/cjowen1/catomistic.git
cd catomistic
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest
```

5. Run tests with coverage:
```bash
pytest --cov=catomistic
```

6. Build documentation:
```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

MIT License 
