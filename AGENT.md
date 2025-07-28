# PSFormer Agent Guide

## Test Commands
- **Run all tests**: `pytest tests/ -v`
- **Run single test file**: `pytest tests/test_ps_block.py -v`
- **Run with coverage**: `pytest tests/ --cov=src --cov-report=term-missing`
- **Run specific test class**: `pytest tests/test_ps_block.py::TestPSBlockDataValidation -v`

## Architecture
- **Core components**: PSBlock (parameter shared), RevIN (normalization), two-stage segment attention
- **Structure**: `src/blocks/` contains atomic units, `tests/` contains comprehensive test suites
- **Data flow**: Raw time series → RevIN norm → patching → segments → PSformer encoder → output → RevIN inverse
- **Key innovation**: Parameter sharing across all PS blocks within encoder layers

## Code Style
- **Dependencies**: PyTorch, pytest, numpy (see requirements.txt)
- **Imports**: Standard library first, then torch/nn, then local imports
- **Error handling**: Explicit validation for tensor shapes, NaN/inf values, dimension mismatches
- **Testing**: Comprehensive test classes covering data validation, processing, behavior, robustness
- **Docstrings**: Include Args/Returns sections, reference paper equations where applicable
- **Naming**: snake_case for variables/functions, PascalCase for classes, descriptive names
- **Type hints**: Use torch.Tensor for tensor parameters, int for dimensions
