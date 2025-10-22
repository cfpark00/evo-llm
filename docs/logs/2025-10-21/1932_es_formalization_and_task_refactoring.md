# 1932 - ES Formalization and Task Refactoring

**Date:** 2025-10-21
**Time:** 19:32
**Session Duration:** Full day continuation from context-limited session

## Summary

Formalized the Evolution Strategies (ES) fine-tuning framework from scratch scripts into production code following repository conventions. Major work included removing hardcoded values, refactoring task-specific logic into classes, fixing config-driven architecture, and debugging initialization/perturbation scaling.

---

## Major Work Completed

### 1. ES Training Script Formalization (`src/scripts/es_train.py`)

**Created formal orchestration script following `docs/repo_usage.md` template:**
- Proper config validation with fail-fast error handling
- Uses `init_directory()` from `src/utils.py`
- Copies config.yaml to output_dir for reproducibility
- Creates standard subdirectories: `figures/`, `results/`, `logs/`, `loras/`
- Removed excessive try-except blocks per user's fail-fast philosophy
- Only one try-except for server cleanup in finally block

**Key architectural decisions:**
- SGLang server launched ONCE at start with dummy LoRA for shape
- Dynamic LoRA load/unload instead of server restarts
- Parallel evaluation using ThreadPoolExecutor
- Multi-LoRA batching (--max-loras-per-batch parameter)

### 2. Configuration System

**Created `configs/experiments/es_conciseness.yaml`:**
```yaml
output_dir: "data/es_conciseness_v1"
seed: 33

task:
  name: "conciseness"
  train_data: "data/tasks/conciseness/conciseness_train.json"
  test_data: "data/tasks/conciseness/conciseness_test.json"

model:
  name: "Qwen/Qwen2.5-7B-Instruct"

lora:
  rank: 1
  alpha: 2
  target_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

es:
  num_generations: 30
  population_size: 30
  learning_rate: 0.01
  noise_multiplier: 0.1

evaluation:
  max_tokens: 100
  temperature: 0.0
```

**Created `configs/experiments/es_countdown.yaml`:**
- Same structure but `max_tokens: 1024` for longer countdown responses
- Single data file that gets split 80/20 at runtime

**Removed hardcoded values:**
- LoRA rank, alpha, target_layers, target_modules now from config
- Noise multiplier (absorbed the 0.1 factor)
- Model name and temperature now from config
- Seed added to config (default 42)

### 3. Data Organization

**Reorganized task data:**
- Created `data/tasks/conciseness/` directory
- Moved conciseness train/test JSONs there
- Created `data/tasks/countdown/` directory
- Copied countdown.json from paper repo

**Data structure:**
```
data/tasks/
├── conciseness/
│   ├── conciseness_train.json (2 examples)
│   └── conciseness_test.json (8 examples)
└── countdown/
    └── countdown.json (23026 lines, split 80/20 at runtime)
```

### 4. Task Class Refactoring

**Created `src/tasks/` module with class-based architecture:**

**`src/tasks/__init__.py`:**
- Task registry mapping task names to classes
- `get_task(task_name, task_config)` function

**`src/tasks/conciseness.py`:**
```python
class ConcisenessTask:
    def __init__(self, config): ...
    def load_data(self, seed=None): ...
    def compute_reward(self, generated_text, example):
        target_text = example['target']
        return -abs(len(generated_text) - len(target_text))
```

**`src/tasks/countdown.py`:**
```python
class CountdownTask:
    def __init__(self, config): ...
    def load_data(self, seed=None):
        # Loads full dataset and splits 80/20 with seed
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(full_data))
        # ... split logic
    def compute_reward(self, generated_text, example):
        # Format reward (0.1 weight) + Answer reward (1.0 weight)
        return format_reward * 0.1 + answer_reward
    def _format_reward(self, response, end_token=None): ...
    def _answer_reward(self, response, numbers, target): ...
```

**Benefits of class-based approach:**
- Complete isolation of task-specific logic
- Consistent interface across tasks
- Easy to add new tasks
- Encapsulates data loading and reward computation

### 5. LoRA Initialization Optimization

**Problem identified:** Loading base model 30 times for initial population (wasteful)

**Solution implemented:**
```python
def create_initial_population(model_name, output_dir, population_size, per_layer_noise_scales, lora_config):
    # Load base model ONCE
    model = AutoModelForCausalLM.from_pretrained(...)
    peft_model = get_peft_model(model, lora_config)

    # Randomize weights and save 30 times
    for i in range(population_size):
        randomize_lora_weights(peft_model, per_layer_noise_scales, target_layers)
        peft_model.save_pretrained(lora_path)

    del model, peft_model
```

### 6. Noise Multiplier Absorption

**Changed from:**
```python
per_layer_noise_scales[layer_num] = mean_magnitude * 0.1 * noise_multiplier
# Config: noise_multiplier: 1.0
```

**To:**
```python
per_layer_noise_scales[layer_num] = mean_magnitude * noise_multiplier
# Config: noise_multiplier: 0.1
```

**Applied consistently to:**
- Per-layer noise scale computation
- Random initialization (`randomize_lora_weights`)
- Perturbation (`perturb_lora`)

### 7. Visualization Updates

**Changed plot from mean ± std to percentiles:**
```python
# Old: std can go beyond actual data range
plt.fill_between(generations,
                 np.array(mean_rewards) - np.array(std_rewards),
                 np.array(mean_rewards) + np.array(std_rewards), ...)

# New: 20th-80th percentile range
p20_rewards = [np.percentile(h['rewards'], 20) for h in evolution_history]
p80_rewards = [np.percentile(h['rewards'], 80) for h in evolution_history]
plt.fill_between(generations, p20_rewards, p80_rewards, ...)
```

**Plot now overwrites same file:**
- Changed from `evolution_gen{N}.png` to `evolution.png`
- Updates in place each generation

### 8. Bash Scripts

**Created:**
- `scripts/experiments/train_es_conciseness.sh`
- `scripts/experiments/train_es_countdown.sh`

**Format:**
```bash
#!/bin/bash
uv run python src/scripts/es_train.py configs/experiments/es_conciseness.yaml "$@"
```

Made executable with `chmod +x`

---

## Key Bugs Fixed

### 1. Missing LoRA API Parameters
- **Error:** Used `"model": lora_name` instead of `"lora_path": lora_name`
- **Fix:** Corrected API parameter name
- **User feedback:** "do you want to die?"

### 2. Hardcoded Model Name and Temperature
- **Problem:** `generate()` method had hardcoded values
- **Fix:** Pass from config to `SGLangLoRAEvaluator.__init__()`

### 3. Wrong LoRA Directory Path
- **Problem:** Used wrong path from scratch script
- **Fix:** Updated to correct lora_adapters path

### 4. Undefined Seed Variable
- **Error:** `NameError: name 'seed' is not defined`
- **Fix:** Added `seed = config.get('seed', 42)` before task loading

### 5. PEFT Config Warning
- **Warning:** "Already found a `peft_config` attribute in the model"
- **Root cause:** Trying to reuse model object for multiple LoRAs
- **Fix:** Create PEFT model once, randomize weights multiple times

---

## Performance Investigation

**Issue discovered:** Performance worse than scratch implementation

**Root cause analysis:**
```
Scratch version:
- Initial LoRAs: init_scale = 20.0 multiplier on A and B matrices
- Perturbation: noise_scale = 0.10 (hardcoded per-layer values)

Current version:
- Initial LoRAs: N(0, mean_magnitude * 0.1) - NO 20x multiplier
- Perturbation: noise_scale = mean_magnitude * 0.1

Result: Initial LoRAs 20x weaker, less signal for ES to work with
```

**User action:** Will try increasing noise_multiplier in config

---

## File Structure Changes

### New Directories
```
src/tasks/                    # Task-specific logic
data/tasks/conciseness/       # Conciseness task data
data/tasks/countdown/         # Countdown task data
configs/experiments/          # Experiment configs
scripts/experiments/          # Experiment runner scripts
```

### New Files
```
src/tasks/__init__.py
src/tasks/conciseness.py
src/tasks/countdown.py
src/scripts/es_train.py
configs/experiments/es_conciseness.yaml
configs/experiments/es_countdown.yaml
scripts/experiments/train_es_conciseness.sh
scripts/experiments/train_es_countdown.sh
data/tasks/conciseness/conciseness_train.json
data/tasks/conciseness/conciseness_test.json
data/tasks/countdown/countdown.json
```

### Modified Files
- Removed task data from old `data/conciseness/` location
- Updated all imports and paths

---

## Technical Details

### ES Algorithm Implementation

**Per-layer noise scaling:**
```python
# Analyze base model weights
for layer_num in target_layers:
    layer_magnitudes = []
    for name, param in base_model.named_parameters():
        if f"layers.{layer_num}." in name and any(t in name for t in target_modules):
            layer_magnitudes.append(param.abs().mean().item())

    mean_magnitude = np.mean(layer_magnitudes)
    per_layer_noise_scales[layer_num] = mean_magnitude * noise_multiplier
```

**Initial population creation:**
- Uses PEFT's `init_lora_weights=False`
- Manually initializes both A and B matrices with per-layer scaled normal distribution
- Avoids PEFT's default (A random, B zeros)

**Perturbation:**
- Loads base LoRA weights
- Adds Gaussian noise scaled by per-layer magnitudes
- Saves perturbed weights as new LoRA

**ES update:**
- Normalized rewards (z-score)
- Weighted average of perturbations
- Learning rate applied to update

### Parallel Evaluation

**Multi-LoRA batching confirmed working:**
- SGLang batches requests across different LoRAs
- 4 requests with 2 LoRAs: 2.34s (4x speedup vs sequential)
- Uses `--max-loras-per-batch=30` parameter

**Thread pool evaluation:**
```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(eval_one_lora, lora_name): lora_name
               for lora_name in lora_names}
    for future in as_completed(futures):
        lora_name, mean_reward, results = future.result()
```

---

## User Preferences Applied

1. **Fail-fast philosophy:** No silent error handling, removed all try-except except cleanup
2. **No hardcoded values:** Everything moved to config
3. **No invented data:** Used actual countdown.json from paper repo
4. **Class-based tasks:** Complete isolation of task logic
5. **Percentiles over std:** Plot shows 20th-80th percentile range
6. **Single plot file:** Overwrites `evolution.png` instead of `evolution_gen{N}.png`
7. **Seed-based splitting:** Countdown data split deterministically at runtime

---

## Hyperparameters from Paper

**Conciseness:**
- NUM_ITERATIONS: 1000
- POPULATION_SIZE: 30
- SIGMA: 0.001
- ALPHA: 0.0005
- max_new_tokens: 100

**Countdown:**
- NUM_ITERATIONS: 1000
- POPULATION_SIZE: 30
- SIGMA: 0.001
- ALPHA: 0.0005
- max_new_tokens: 1024 (only difference)

**Our current config:**
- num_generations: 30 (shorter for testing)
- population_size: 30 ✓
- learning_rate: 0.01 (20x higher than paper!)
- noise_multiplier: 0.1 (100x higher than paper's SIGMA!)

---

## Next Steps

1. Test with increased noise_multiplier to match scratch performance
2. Consider adding 20x initialization scaling as separate config parameter
3. Run full experiments with paper hyperparameters
4. Compare performance between conciseness and countdown tasks

---

## Code Quality Notes

**Followed repository conventions:**
- All Python code in `src/`
- All configs in `configs/`
- All runner scripts in `scripts/`
- Standard output structure: `data/{experiment}/figures|results|logs|loras/`
- Config copying for reproducibility
- Proper imports using `from src.module import ...`

**Clean architecture:**
- Task logic isolated in task classes
- Evaluation logic generic across tasks
- ES algorithm task-agnostic
- SGLang interface abstracted in evaluator class
