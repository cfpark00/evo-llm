# Research Context: Evolution Strategies for LLM Fine-tuning

## Project Overview

This project explores **Evolution Strategies (ES)** as an alternative to Reinforcement Learning (RL) for fine-tuning Large Language Models (LLMs). Recent work (Qiu et al., 2025) demonstrated that weight perturbation methods can discover strategies that RL cannot find, achieving superior performance on various benchmarks.

## Research Goal

**Primary Goal**: Develop an efficient ES framework using SGLang and LoRA adapters to perform evolutionary fine-tuning of LLMs, and deeply understand how ES differs from RL in the context of LLM optimization.

## Key Innovation: LoRA-based ES

Unlike the original paper which perturbs full model parameters (billions of parameters), our approach:
- **Represents each population member as a LoRA adapter** (low-rank perturbation)
- **Leverages SGLang for efficient multi-LoRA serving** (parallel evaluation)
- **Workflow**:
  1. Sample N LoRA adapters (population members)
  2. Load and evaluate all LoRAs in parallel via SGLang
  3. Compute weighted average LoRA based on fitness
  4. Generate next generation of LoRA perturbations
  5. Repeat

This makes ES tractable for research-scale compute while maintaining the core evolutionary dynamics.

## Current Task: Conciseness

**Primary Task**: Conciseness optimization (from paper)
- **Goal**: Generate responses with length matching target examples
- **Reward**: -|len(generated) - len(target)| (negative length difference)
- **Training data**: 2 examples (paper-compliant minimal supervision)
- **Test data**: 8 examples (for generalization evaluation)
- **Rationale**: Paper showed ES vs RL differences most clearly on this task

## Target Task Domains (Future)

Tasks where ES is hypothesized to excel over RL:

1. **Reasoning tasks** (e.g., Countdown, mathematical reasoning)
   - Long-horizon, sparse rewards
   - Credit assignment is difficult for RL

2. **Multi-turn dialogue tasks**
   - Sequential decision-making over extended conversations
   - Requires consistent behavior across turns

3. **High-dimensional policy tasks**
   - Complex action spaces
   - Where parameter-space exploration may be advantageous

4. **Deliberate problem-solving**
   - Tasks requiring step-by-step reasoning
   - Process rewards are difficult to define

## Model Selection

- **Primary model**: Qwen2.5-7B-Instruct
- **Rationale**: Balance between capability and computational feasibility
- **Future**: May extend to other model families/sizes for robustness testing

## ES Algorithm Details

### Core Algorithm (Simplified NES variant)
Based on the paper's approach (Algorithm 2), but with LoRA instead of full parameters:

```
For each generation t:
  1. Sample N LoRA perturbations from base model
  2. Evaluate each LoRA on task (parallel via SGLang)
  3. Normalize rewards (z-score)
  4. Weighted update: θ_t = θ_{t-1} + α * (1/N) * Σ(R_n * ε_n)
  5. Generate next population from θ_t
```

### Hyperparameters (Current Configuration)
- **Population size (N)**: 30 (following paper)
- **Learning rate (α)**: 0.01 (higher than paper's 5×10⁻⁴ due to LoRA)
- **Noise multiplier**: 1.0x (applied to per-layer base noise)
- **LoRA rank**: 1 (minimal, fast)
- **LoRA alpha**: 2
- **LoRA init scale**: 20x (both A and B matrices)
- **Target layers**: 0-9 (first 10 layers)
- **Generations**: 5 (testing; increase to 50-100 for full runs)

### No Enhancements (for now)
We deliberately avoid the paper's enhancements to keep the algorithm simple:
- ❌ Rank transformation of rewards
- ❌ Mirrored sampling
- ❌ Weight decay
- ❌ Virtual batch normalization
- ❌ Adam optimizer (using vanilla gradient update)

This allows us to isolate the core ES dynamics and understand the fundamental differences from RL.

## Current State

### Completed
✅ Read and understood the ES fine-tuning paper (Qiu et al., 2025)
✅ Set up project structure following lab conventions
✅ Tested Countdown task setup and reward functions
✅ Created test script for Qwen2.5-7B-Instruct with chat format
✅ Initialized and saved LoRA adapters (30 adapters, rank 1, alpha 2, layers 0-9)
✅ Researched SGLang's LoRA capabilities:
  - Multi-LoRA batching is supported
  - Dynamic loading/unloading is implemented but not fully documented
  - Chat completions API works with LoRA adapters

✅ Built complete ES framework with all core components:
  - SGLang evaluator class with health checks
  - LoRA weight loading/saving utilities
  - Proper ES update with explicit learning rate (α=0.01)
  - Per-layer noise scaling based on weight magnitudes
  - Gaussian perturbation function
  - Evolution loop (configurable generations)
  - Results tracking and persistence
  - Real-time visualization of evolution progress
  - Test set evaluation after evolution completes

✅ **Replaced Countdown task with Conciseness task**:
  - Training data: 2 examples (paper-compliant)
  - Test data: 8 examples (for generalization evaluation)
  - Reward: -|len(generated) - len(target)| (conciseness metric)

✅ **Fixed critical LoRA API bug**:
  - Problem: LoRAs weren't being used (all outputs identical)
  - Root cause: Wrong API parameter (`"model": lora_name`)
  - Solution: Use `"lora_path": lora_name` in chat completions
  - Status: Verified working with test script

✅ **Fixed LoRA initialization for ES**:
  - Problem: PEFT default has B matrix = zeros (no initial diversity)
  - Solution: Initialize both A and B matrices randomly
  - Scaling: 20x multiplier on both matrices for stronger signal
  - Verification: Checked saved LoRA, B matrix std=0.40 (non-zero)

✅ **Implemented per-layer noise scaling**:
  - Analyzed base model weight magnitudes for layers 0-9
  - Computed 10% of mean magnitude per layer as base noise
  - Applied configurable multiplier (currently 1.0x)
  - Each layer gets appropriate noise scale

### Critical Blockers - ALL RESOLVED ✅

~~**SGLang Dynamic LoRA Loading**~~ → FIXED
- **Problem**: LoRAs with explicit layer paths incompatible with SGLang
- **Solution**: Use PEFT's `layers_to_transform` parameter instead of explicit module paths
- **Details**: See `docs/logs/2025-10-21/1452_sglang_lora_fix.md`

~~**LoRA Not Being Applied During Generation**~~ → FIXED
- **Problem**: All LoRAs produced identical outputs regardless of weights
- **Root cause**: Using wrong API parameter in SGLang chat completions
- **Solution**: Changed from `"model": lora_name` to `"lora_path": lora_name`
- **Details**: See `docs/logs/2025-10-21/1748_es_conciseness_task_complete_setup.md`

~~**LoRA Initialization Lacks Diversity**~~ → FIXED
- **Problem**: All initial LoRAs identical (B matrix = zeros by PEFT default)
- **Solution**: Initialize both A and B matrices with random normal + 20x scaling
- **Details**: See `docs/logs/2025-10-21/1748_es_conciseness_task_complete_setup.md`

### Next Steps
1. **Run first complete ES experiment**:
   - Execute full 5-generation run with population size 30
   - Monitor evolution plots for improvement signal
   - Evaluate best LoRA on test set
   - Compare evolved LoRA vs base model performance

2. **Hyperparameter tuning** (if needed):
   - Adjust noise multiplier based on results (currently 1.0x)
   - Adjust learning rate α (currently 0.01)
   - Increase generations (5 → 50-100) for full convergence

3. **Move to production structure**:
   - Extract ES framework → src/es_framework.py
   - Create orchestration script → src/scripts/es_train.py
   - Create config → configs/experiments/conciseness_es.yaml
   - Create runner → scripts/experiments/train_conciseness_es.sh

4. **Analysis framework** (future):
   - Track metrics: reward, KL divergence, parameter magnitude shifts
   - Compare ES vs RL behavior
   - Analyze exploration patterns

## Research Questions & Hypotheses

### Primary Questions
1. **Can LoRA-based ES achieve similar performance to full-parameter ES?**
   - Hypothesis: Yes, if LoRA rank is sufficient to capture necessary perturbations

2. **How does ES differ from RL in terms of**:
   - Sample efficiency
   - Robustness to hyperparameters
   - Reward hacking tendency
   - Consistency across runs
   - KL divergence from base model

3. **What makes ES effective in parameter space despite high dimensionality?**
   - Hypothesis: LLMs have low intrinsic dimensionality (Aghajanyan et al., 2021)
   - Hypothesis: ES's Gaussian smoothing makes reward landscape more navigable

4. **When does ES outperform RL, and why?**
   - Long-horizon sparse rewards (Countdown task)
   - Smaller base models (where RL fails to bootstrap)
   - Tasks prone to reward hacking

### Open-Ended Exploration
- What is the minimal LoRA rank needed for effective ES?
- How does population size affect convergence and diversity?
- Can we visualize the evolution trajectory in LoRA space?
- Does ES find fundamentally different solutions than RL?
- How do different noise scales affect exploration vs exploitation?

## Key References

**Main Paper**:
- Qiu, X., Gan, Y., Hayes, C. F., et al. (2025). "Evolution Strategies at Scale: LLM Fine-tuning Beyond Reinforcement Learning." arXiv:2509.24372

**Key Claims from Paper**:
- ES is more sample efficient than RL (needs <20% of samples)
- ES works with small populations (N=30) despite billions of parameters
- ES is robust across different base LLMs
- ES avoids reward hacking without KL penalty
- ES produces more consistent results across runs
- ES discovers dominant Pareto front (reward vs KL divergence)

**Supporting Literature**:
- Salimans et al. (2017): OpenAI ES for deep RL
- Lehman et al. (2018): ES optimizes solution distributions, not single solutions
- Aghajanyan et al. (2021): Low intrinsic dimensionality of LLMs

## Success Criteria

### Milestone 1: Basic Implementation ✅ COMPLETE
- [x] Understand task setup (Conciseness)
- [x] Test base model inference
- [x] Create LoRA adapters with proper initialization
- [x] Implement complete ES loop with LoRA
- [x] Fix critical LoRA API bug
- [x] Add per-layer noise scaling
- [x] Add visualization and test evaluation
- [ ] **NEXT**: Run full experiment and verify convergence

### Milestone 2: Reproduce Paper Results (with LoRA)
- [ ] Match paper's Countdown task accuracy
- [ ] Validate ES > RL on at least one task
- [ ] Measure sample efficiency

### Milestone 3: Deep Understanding
- [ ] Characterize ES vs RL differences quantitatively
- [ ] Identify conditions where ES excels
- [ ] Understand mechanisms (parameter space smoothing, solution distribution optimization)

### Milestone 4: Novel Contributions (Open-ended)
- [ ] Discover new insights about LoRA-based ES
- [ ] Test hypotheses about when/why ES works
- [ ] Explore new task domains
- [ ] Optimize efficiency/scalability

## Technical Notes

### SGLang + LoRA Infrastructure
- **Server startup**: `--lora-paths lora0=path/to/adapter1 lora1=path/to/adapter2`
- **Multi-LoRA batching**: Up to 8 adapters per batch (configurable)
- **Backend**: Triton (recommended over FlashInfer)
- **Chat API**: Use `model="lora0"` parameter to select adapter
- **Dynamic loading**: Implemented in v0.4.10+ but API not fully documented
- **Radix cache**: Must be disabled when using LoRA (`--disable-radix-cache`)

### LoRA Configuration
- **Rank**: 1 (minimal, fast, test if sufficient)
- **Alpha**: 2
- **Target modules**: All attention + MLP projections in layers 0-10
  - `q_proj, k_proj, v_proj, o_proj` (attention)
  - `gate_proj, up_proj, down_proj` (MLP)
- **77 total target modules** (7 modules/layer × 11 layers)

### Computational Considerations
- **Memory**: LoRA adapters are ~1000x smaller than full model
- **Parallelism**: Can evaluate entire population in parallel
- **Storage**: Each LoRA checkpoint is small (~few MB vs multi-GB full model)
- **Inference-only**: No backpropagation needed (ES uses forward passes only)

## Key Findings & Insights

### Critical Bug Fixes (2025-10-21)
1. **LoRA API parameter**: Must use `"lora_path": lora_name` in chat completions, not `"model": lora_name`
   - Without this fix, LoRAs aren't applied at all (base model always used)
   - Verified by comparing outputs with/without LoRA specification

2. **LoRA initialization for ES**: PEFT's default initialization (B=zeros) is inappropriate for ES
   - Default ensures LoRA has zero initial effect on model
   - For ES, we need initial population diversity
   - Solution: Initialize both A and B matrices randomly with 20x scaling

3. **Per-layer noise scaling**: Different layers have different weight magnitudes
   - Analyzed base model weights to compute appropriate noise per layer
   - Using 10% of mean weight magnitude as base noise scale
   - Applies configurable multiplier (currently 1.0x) for tuning

## Open Questions / Unclear Areas

1. **LoRA expressiveness**: Is rank 1 sufficient, or do we need higher rank?
   - Current: rank 1 for speed
   - Need to verify if this is expressive enough for ES to find improvements

2. **Hyperparameter tuning**:
   - Learning rate α=0.01 may need adjustment based on results
   - Noise multiplier 1.0x may be too small or too large
   - Need to run experiments to calibrate

3. **Evaluation protocol**: Currently using greedy decoding (deterministic)
   - Matches paper's approach
   - Ensures reproducible rewards

## Notes
- This is a research exploration project, not a production system
- We prioritize understanding over performance optimization
- Fail-fast philosophy: explicit failures reveal insights
- All experiments should be reproducible via configs in `configs/`
- Document insights and dead-ends in `docs/logs/YYYY-MM-DD/`

---

**Last Updated**: 2025-10-21 17:48
**Status**: Full ES framework complete with all critical bugs fixed. Ready to run first complete experiment.
