# CLAUDE.md - Development Guidelines

This document outlines the project structure and development conventions.

## Project Structure

### `/scripts/` - Execution Scripts
All experiments, data creation, and analysis must be run through minimalistic bash scripts in this directory.
- Scripts should be simple wrappers that call Python code with appropriate configs
- Use `uv run python src/<module>.py configs/<config>.yaml` pattern
- Make scripts executable with `chmod +x`
- **It is advised to organize with subfolders** for different purposes based on your project needs
- Example subfolders (adapt to your project):
  - `scripts/data_generation/`
  - `scripts/training/`
  - `scripts/evaluation/`
  - `scripts/experiments/`

### `/src/` - Python Source Code
All Python code must be in this directory.
- No Python files in root or other directories
- Use clear module names that describe functionality
- Import between modules using `from src.module import ...`
- `src/utils.py` - For stateless utility functions expected to be used repetitively throughout the research
- `src/scripts/` - For orchestration scripts (entry points) that coordinate experiments

### `/configs/` - Configuration Files
All configuration must be in YAML files in this directory.
- Every run takes exactly one config file as input
- Config files must include `output_dir` field specifying where results go
- **It is advised to organize with subfolders** by purpose based on your project needs
- Example subfolders (adapt to your project):
  - `configs/data/`
  - `configs/experiments/`
  - `configs/models/`
  - `configs/training/`
- Use descriptive names for your configs
- Structure:
  ```yaml
  output_dir: "data/experiment_name"
  experiment:
    epochs: 100
    num_runs: 3
  # ... other config
  ```

### `/data/` - Output Directory
All experiment outputs go here, organized by `output_dir` from configs.
- This directory is gitignored
- Each run creates subdirectories: `figures/`, `results/`, `logs/`
- Directory structure is auto-created by the code

### `/scratch/` - Temporary Work Directory
Freeform directory for temporary tests, one-off scripts, and experimental code.
- Everything except `.gitkeep` is gitignored
- Use for quick tests, debugging, temporary files
- DO NOT put important code here - it won't be tracked by git
- Perfect for trying things out before integrating into the main codebase

### `/docs/` - Documentation
Contains notes and logs from and for developers.
- `/docs/logs/YYYY-MM-DD/` - Daily development logs
- `/docs/structure.txt` - Project structure overview
- Technical notes and analysis

## Development Workflow

### Package Management
We use `uv` for Python package management.

**Always use the uv virtual environment:**
```bash
# Option 1: Activate the environment
source .venv/bin/activate
python src/train.py configs/experiment_config.yaml

# Option 2: Use uv run
uv run python src/train.py configs/experiment_config.yaml
```

**Always add packages with uv:**
```bash
uv add numpy pandas matplotlib
uv add scipy  # Never use pip install
```

### Running Experiments

1. Create or modify a config file in `configs/`
2. Run via script in `scripts/`:
   ```bash
   bash scripts/run_experiment.sh
   ```
3. Results appear in the `output_dir` specified in config

### Adding New Experiments

1. Create new config: `configs/new_experiment.yaml`
2. Add Python code in `src/` if needed
3. Create runner script: `scripts/run_new_experiment.sh`
   ```bash
   #!/bin/bash
   uv run python src/module.py configs/new_experiment.yaml
   ```
4. Make executable: `chmod +x scripts/run_new_experiment.sh`

## Code Conventions

### Imports
```python
# Good - using src module structure
from src.models import BaseModel
from src.utils import DataLoader

# Bad - relative imports or files outside src/
from models import BaseModel  # Wrong
from ../utils import helper   # Wrong
```

### Config Loading
```python
# Every main script should accept config path
if __name__ == "__main__":
    import sys
    import yaml
    from pathlib import Path
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use config['output_dir'] for all outputs
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
```

### Output Organization
```python
# Standard output structure
output_dir/
├── figures/      # Plots and visualizations
├── results/      # JSON/CSV result files  
└── logs/         # Logs and debug info
```

## Environment Setup

1. Clone repository
2. Copy `.env.example` to `.env` and configure settings
3. Install dependencies: `uv sync`
4. Run experiments: `bash scripts/run_experiment.sh`

## Git Conventions

- `/data/` is gitignored - never commit experiment outputs
- `.env` is gitignored - use `.env.example` as template
- `uv.lock` is committed for reproducibility
- Commit messages should be descriptive

## Testing

Before committing:
1. Ensure all scripts run without errors
2. Check that output appears in correct directories
3. Verify configs have required `output_dir` field
4. Run `uv sync` to ensure dependencies are locked

## Common Commands

```bash
# Run experiment
bash scripts/run_experiment.sh

# Add new package
uv add package_name

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Troubleshooting

- **ModuleNotFoundError**: Ensure you're using `uv run` or activated `.venv`
- **Import errors**: Check that imports use `from src.module import ...`
- **Missing output**: Verify config has `output_dir` field
- **Permission denied**: Make scripts executable with `chmod +x`

## Project Information

### Research Focus
[Add your research questions and goals here]

### Key Components
[Add descriptions of your main modules and components here]

### Running Experiments
[Add specific instructions for your experiments here]

### Key Metrics
[Add the metrics you're tracking here]

### Expected Outcomes
[Add your expected results here]

---

**Note**: This document covers the basics. For more detailed information about development practices, code organization, and practical guidelines, see `docs/repo_usage.md`