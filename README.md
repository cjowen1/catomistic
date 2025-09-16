# catomistic

A package for atomistic simulations and machine learning in catalysis and surface chemistry. This code is actively being developed, as of May 07, 2025, by Cameron J. Owen.

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

3. Run tests (not yet implemented):
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
5. Run tests and ensure they pass
6. Format and lint with ruff
7. Submit a pull request

## License

MIT License 
