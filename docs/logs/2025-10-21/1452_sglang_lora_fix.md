# Development Log: SGLang LoRA Integration Fix

**Date**: 2025-10-21
**Time**: 14:52
**Session Duration**: ~2.5 hours

## Summary

Fixed critical SGLang LoRA compatibility issues. Discovered that layer-specific LoRA configurations (using explicit layer paths) were incompatible with SGLang. Solved by using PEFT's `layers_to_transform` parameter instead. Successfully validated dynamic LoRA loading works with layers 0-9.

## Problem Statement

SGLang server was crashing when trying to load LoRA adapters:
- **Error**: `NotImplementedError` in `get_hidden_dim()` during LoRA memory pool initialization
- **Root cause**: LoRAs created with explicit layer paths in `target_modules` (e.g., `model.layers.0.self_attn.q_proj`) were incompatible with SGLang's LoRA manager

## Investigation Process

### Initial Attempts (Failed)
1. **Tried**: Specifying `--lora-target-modules all`
   - **Result**: Still crashed with same error

2. **Tried**: Using `--lora-backend csgmv` instead of `triton`
   - **Result**: Same crash

3. **Tried**: Preloading one LoRA with `--lora-paths` to infer config
   - **Result**: Server crashed during scheduler initialization

4. **Tried**: Manually specifying all target module names
   - **Result**: Still incompatible with layer-specific paths

### Breakthrough: Isolated Testing

Created `scratch/sglang_lora_only_test/` for systematic testing:

#### Test 1: Create LoRA (layers 0-9)
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=1,
    lora_alpha=2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    layers_to_transform=list(range(0, 10)),  # KEY: Use this instead of explicit paths
    lora_dropout=0.0,
    bias="none",
    inference_mode=False,
)
```

**Result**: ✅ LoRA created successfully (901,120 trainable params, 0.0118%)

#### Test 2: Load LoRA at Server Startup
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --lora-paths testlora=<path> \
    --max-loras-per-batch 1 \
    --lora-backend triton \
    --disable-radix-cache
```

**Result**: ✅ Server started successfully, inference worked!

#### Test 3: Dynamic LoRA Loading
```python
# Preload dummy, then unload
requests.post(url + "/unload_lora_adapter", json={"lora_name": "dummy"})

# Dynamically load new LoRA
requests.post(url + "/load_lora_adapter",
              json={"lora_name": "newlora", "lora_path": path})
```

**Result**: ✅ Dynamic loading worked perfectly!

## Solution

### Key Insight
SGLang expects LoRAs created with **simple module patterns** + `layers_to_transform`, NOT explicit layer paths.

**Wrong** (SGLang incompatible):
```python
target_modules = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    # ... explicit paths for each layer
]
```

**Correct** (SGLang compatible):
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
layers_to_transform = list(range(0, 10))  # Specify layers here
```

### Updated Files

**`init_loras.py`**:
- Changed LoRA config to use `layers_to_transform=list(range(0, 10))`
- Removed explicit layer path construction
- Fixed JSON serialization error (convert set to list)

**`es_test_run.py`**:
- Server launch now preloads one LoRA for config inference
- Simplified server args (removed manual `--lora-target-modules`)
- Unloads dummy LoRA before starting generation 0

## Technical Details

### LoRA Configuration
- **Rank**: 1
- **Alpha**: 2
- **Target modules**: All attention (q/k/v/o_proj) + MLP (gate/up/down_proj)
- **Layers**: 0-9 (10 layers total, out of 28 in Qwen2.5-7B)
- **Trainable params**: ~901K per adapter (~0.012% of 7.6B total)

### SGLang Server Config
```bash
--model-path Qwen/Qwen2.5-7B-Instruct
--lora-paths dummy=<initial_lora_path>  # Infer config from this
--max-loras-per-batch 8
--lora-backend triton
--disable-radix-cache
```

### Dynamic Loading Workflow
1. Server starts with dummy LoRA (for config inference)
2. Unload dummy via `/unload_lora_adapter`
3. Dynamically load new LoRAs via `/load_lora_adapter`
4. Use in inference via `/v1/chat/completions` with `model=<lora_name>`

## Validation Results

**Static Loading**: ✅ Works
**Dynamic Loading**: ✅ Works
**Inference with LoRA**: ✅ Works
**Layers 0-9 targeting**: ✅ Works

## Next Steps

1. ✅ Create 5 LoRA adapters with new config (`run_init.sh`)
2. ⏳ Run full ES loop (`run_es_test.sh`)
3. ⏳ Validate evolution across generations
4. ⏳ Measure performance on Countdown task

## Files Created/Modified

### Created
```
scratch/sglang_lora_only_test/
├── test1_create_lora.py       # LoRA creation test
├── test2_launch_server.py     # Server launch test
├── test3_dynamic_load.py      # Dynamic loading test
└── test_lora/                 # Working LoRA (layers 0-9)
```

### Modified
```
scratch/sglang_lora_test/
├── init_loras.py              # Fixed LoRA config (layers_to_transform)
└── es_test_run.py             # Simplified server launch
```

## Lessons Learned

1. **PEFT vs SGLang compatibility**: SGLang has specific expectations for LoRA structure
2. **Isolated testing is critical**: Breaking down the problem into 3 simple tests revealed the issue quickly
3. **`layers_to_transform` is the key**: This PEFT parameter is compatible with SGLang's layer indexing
4. **Dynamic loading works**: No need to restart server between generations

## Time Breakdown

- Initial debugging (failed approaches): ~1.5 hours
- Isolated testing setup: ~30 min
- Solution implementation: ~20 min
- Validation: ~10 min

**Total**: ~2.5 hours

## References

- SGLang LoRA docs: Multi-LoRA serving with dynamic loading
- PEFT LoraConfig: `layers_to_transform` parameter
- SGLang issue: Layer-specific module paths not supported in LoRA memory pool

---

**Status**: ✅ Issue resolved. Ready to run full ES evolution.

**Confidence**: High - all three test cases passed, approach validated.
