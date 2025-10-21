# Development Log: ES Conciseness Task - Complete Setup and Critical Fixes

**Date**: 2025-10-21
**Time**: 17:48
**Session Duration**: ~3 hours

## Summary

Completed full ES evolution setup for conciseness task with critical fixes:
1. Replaced countdown task with conciseness task (paper's task)
2. Fixed LoRA API parameter (`lora_path` not `model`)
3. Fixed LoRA initialization (both A and B matrices now random)
4. Added proper learning rate α=0.01, population size 30
5. Added test set evaluation and plotting after every generation
6. Fixed per-layer noise scaling with 20x multiplier

## Major Changes

### 1. Task Replacement: Countdown → Conciseness

**Created**: `scratch/sglang_lora_test/conciseness_task.py`
- Training data: 2 examples (paper-compliant)
- Test data: 8 examples (for post-evolution eval)
- Reward: `-|len(generated) - len(target)|` (conciseness metric)

**Why**: Paper showed ES vs RL differences most clearly on conciseness task

### 2. Critical Bug Fix: LoRA Not Being Used

**Problem**: All LoRAs produced identical outputs regardless of weights
- Root cause: Using wrong API parameter in SGLang

**Investigation**:
- Created test script comparing base model vs LoRA outputs
- Found outputs were identical → LoRA never applied
- Searched SGLang documentation

**Solution**:
```python
# WRONG (what we had):
{
  "model": lora_name,
  "messages": [...],
}

# CORRECT:
{
  "model": "Qwen/Qwen2.5-7B-Instruct",  # Base model
  "lora_path": lora_name,  # LoRA selector!
  "messages": [...],
}
```

**Fixed in**: `es_test_run.py:53-66`

### 3. LoRA Initialization Fix: B Matrix Was Zero

**Problem**: All initial LoRAs were identical (no diversity in generation 0)
- PEFT default: `lora_A = random, lora_B = zeros`
- This means all LoRAs start with zero effect!

**Solution**: Initialize both A and B randomly
```python
# In init_loras.py:
for name, param in model_with_lora.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
        param.data *= 20.0  # 20x scale for stronger signal
```

**Verification**: Checked saved LoRA - B matrix now has std=0.40 (not zero!)

### 4. ES Algorithm Corrections

**Added explicit learning rate α**:
- Was implicitly α=1.0 (taking full weighted average)
- Now α=0.01 (paper uses 0.0005 for full model, we use higher for LoRA)
- Proper ES update: `θ_new = θ_old + α * advantage_weighted_update`

**Changed function**: `compute_advantage_weighted_average()` → `compute_es_update()`

**Population size**: 5 → 30 (matches paper)

### 5. Per-Layer Noise Scaling

**Enhancement**: Noise proportional to each layer's weight magnitude
- Analyzed base model weight magnitudes per layer (layers 0-9)
- Computed 10% of mean magnitude per layer as base noise
- Applied 20x multiplier for stronger exploration
- Result: Each layer gets appropriate noise scale

**Files**:
- `init_loras.py`: Computes and saves per-layer stats
- `es_test_run.py`: Loads and uses layer-specific noise

### 6. Visualization and Evaluation

**Added plotting after every generation**:
- Plots mean, best, worst rewards over time
- Saved as `evolution_progress_gen{N}.png`
- Allows real-time monitoring of ES progress

**Added test set evaluation**:
- After evolution completes, evaluate best LoRA on 8 test examples
- Measures generalization (not just training performance)
- Saves test results to JSON

## Files Modified

### Created
- `scratch/sglang_lora_test/conciseness_task.py` - Task definition
- `scratch/sglang_lora_test/data/conciseness_train.json` - 2 training examples
- `scratch/sglang_lora_test/data/conciseness_test.json` - 8 test examples
- `scratch/sglang_lora_test/README.md` - Documentation
- `scratch/sglang_lora_only_test/test_lora_working.py` - Debug script

### Modified
- `scratch/sglang_lora_test/init_loras.py`:
  - Increased from 5 to 30 LoRAs
  - Added per-layer weight magnitude analysis
  - Fixed B matrix initialization (random not zero)
  - 20x scaling on both A and B

- `scratch/sglang_lora_test/es_test_run.py`:
  - Replaced countdown → conciseness task
  - Fixed API call: `lora_path` parameter
  - Added proper learning rate (α=0.01)
  - Population size 30
  - Per-layer noise scaling with 20x multiplier
  - Added plotting function
  - Added test set evaluation
  - Train vs test data split

- `scratch/sglang_lora_test/run_es_test.sh`:
  - Updated description with new hyperparameters

## Hyperparameters (Final)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Generations | 5 | Quick test (increase to 50-100 for real) |
| Population | 30 | Matches paper |
| Learning rate (α) | 0.01 | Higher than paper (LoRA needs stronger signal) |
| Noise multiplier | 20x | Applied to per-layer base noise |
| LoRA rank | 1 | Minimal, fast |
| LoRA alpha | 2 | Standard |
| LoRA layers | 0-9 | First 10 layers only |
| LoRA init scale | 20x | Both A and B scaled up |
| Training data | 2 examples | Paper-compliant |
| Test data | 8 examples | For generalization eval |

## Key Insights

1. **LoRA API is tricky**: Parameter name matters (`lora_path` not `model`)
2. **Default PEFT init doesn't work for ES**: B=0 means no diversity
3. **Noise scale matters**: Need to scale up significantly for LoRA (vs full model)
4. **Per-layer noise is better**: Different layers have different magnitudes

## Current Status

✅ **Ready to run full ES evolution experiment**

All infrastructure complete:
- Task implemented correctly (conciseness)
- LoRA initialization creates diverse population
- LoRA API calls work correctly
- ES algorithm has proper learning rate
- Noise scaling calibrated per-layer
- Visualization and evaluation in place

## Next Steps

1. Run full experiment: `bash scratch/sglang_lora_test/run_init.sh && bash scratch/sglang_lora_test/run_es_test.sh`
2. Monitor plots to see if ES is improving
3. Check test set performance for generalization
4. If working, increase generations to 50-100
5. Compare evolved LoRA vs base model on test set

## Technical Debt / TODOs

- [ ] Noise multiplier set to 1.0 by user (was 20x) - may need tuning
- [ ] Only 5 generations - not enough for full convergence
- [ ] Consider saving evolution plots to `data/` not just output dir
- [ ] May need to adjust learning rate based on results
