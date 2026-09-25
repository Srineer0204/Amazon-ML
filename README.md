# Amazon ML Challenge 2026 - Business Entity Resolution

## Project Structure
- `docs/`: Documentation and requirements.
- `src/`: Source code for data processing, feature engineering, and modeling.
- `tests/`: Unit tests for pipeline components.
- `configs/`: Configuration files.
- `experiments/`: Experiment logs and results.
- `models/`: Saved models.
- `outputs/`: Output predictions and submission files.
- `synthetic_data/`: Scripts and data for local synthetic testing.

## Setup
1. Create virtual environment: `python -m venv .venv`
2. Activate virtual environment: `.\.venv\Scripts\Activate.ps1`
3. Install dependencies: `pip install -r requirements.txt`

## Running Tests
To run the synthetic data pipeline test:
```bash
python synthetic_data/run_test_pipeline.py
```

To run unit tests:
```bash
python -m unittest discover tests/
```
