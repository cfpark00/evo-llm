#!/usr/bin/env python3
"""
ES Training Script: Evolution Strategies for LLM Fine-tuning with LoRA

This script performs ES-based fine-tuning using:
- Parallel multi-LoRA evaluation via SGLang
- Dynamic LoRA loading/unloading
- Weighted ES updates based on rewards

Workflow:
1. Initialize population of LoRA adapters
2. Launch SGLang server
3. For each generation:
   - Load population LoRAs
   - Evaluate in parallel
   - Compute ES update
   - Create new generation
   - Unload old, load new
4. Save results and plots
"""

import argparse
import yaml
import json
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from safetensors.torch import save_file, load_file
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import init_directory
from src.tasks import get_task
from sglang.utils import launch_server_cmd, wait_for_server, terminate_process


# ============================================================================
# SGLang Evaluator
# ============================================================================

class SGLangLoRAEvaluator:
    """Handles evaluation of LoRA adapters via SGLang server."""

    def __init__(self, model_name, temperature, base_url="http://localhost"):
        self.base_url = base_url
        self.port = None
        self.model_name = model_name
        self.temperature = temperature

    def set_port(self, port):
        """Set the server port after launch."""
        self.port = port
        self.base_url = f"{self.base_url}:{port}"

    def generate(self, lora_name, messages, max_tokens=512):
        """Generate response using specific LoRA adapter."""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "lora_path": lora_name,
            }
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Handle immediate EOT (None content) - treat as empty string (length 0)
        if content is None:
            return ""
        return content

    def load_lora(self, lora_name, lora_path):
        """Dynamically load a LoRA adapter into the server."""
        response = requests.post(
            f"{self.base_url}/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": str(lora_path)}
        )
        response.raise_for_status()
        return response.json()

    def unload_lora(self, lora_name):
        """Unload a LoRA adapter from the server."""
        response = requests.post(
            f"{self.base_url}/unload_lora_adapter",
            json={"lora_name": lora_name}
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# Evaluation functions
# ============================================================================

def evaluate_single_example(evaluator, lora_name, example, task, max_tokens):
    """Evaluate a single example for a given LoRA."""
    messages = [{"role": "user", "content": example['prompt']}]
    response = evaluator.generate(lora_name, messages, max_tokens=max_tokens)

    # Use task's compute_reward method
    reward = task.compute_reward(response, example)

    return {
        'example_id': example['id'],
        'prompt': example['prompt'],
        'target': example.get('target'),
        'response': response,
        'reward': reward,
        'generated_length': len(response),
    }


def evaluate_lora_on_task(evaluator, lora_name, examples, task, max_tokens):
    """Evaluate a single LoRA on all examples."""
    total_reward = 0
    results = []

    for example in examples:
        result = evaluate_single_example(evaluator, lora_name, example, task, max_tokens)
        total_reward += result['reward']
        results.append(result)

    mean_reward = total_reward / len(examples)
    return mean_reward, results


def evaluate_all_loras_parallel(evaluator, lora_names, examples, task, max_tokens, max_workers):
    """Evaluate all LoRAs in parallel."""
    print(f"\nEvaluating {len(lora_names)} LoRAs in parallel...")
    start_time = time.time()

    def eval_one_lora(lora_name):
        mean_reward, results = evaluate_lora_on_task(evaluator, lora_name, examples, task, max_tokens)
        return lora_name, mean_reward, results

    rewards = [None] * len(lora_names)
    all_results = [None] * len(lora_names)
    lora_name_to_idx = {name: i for i, name in enumerate(lora_names)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_one_lora, lora_name): lora_name
                   for lora_name in lora_names}

        completed = 0
        for future in as_completed(futures):
            lora_name = futures[future]
            _, mean_reward, results = future.result()
            idx = lora_name_to_idx[lora_name]
            rewards[idx] = mean_reward
            all_results[idx] = results
            completed += 1
            print(f"  [{completed}/{len(lora_names)}] {lora_name}: {mean_reward:.4f}")

    elapsed = time.time() - start_time
    print(f"\n✓ Parallel evaluation complete in {elapsed:.2f}s")

    return rewards, all_results


# ============================================================================
# LoRA utilities
# ============================================================================

def load_lora_weights(lora_path):
    """Load LoRA weights from safetensors."""
    lora_path = Path(lora_path)
    weights = load_file(lora_path / "adapter_model.safetensors")

    with open(lora_path / "adapter_config.json", 'r') as f:
        config = json.load(f)

    return config, weights


def save_lora_weights(lora_path, config, weights):
    """Save LoRA weights to safetensors."""
    lora_path = Path(lora_path)
    lora_path.mkdir(parents=True, exist_ok=True)

    save_file(weights, lora_path / "adapter_model.safetensors")

    with open(lora_path / "adapter_config.json", 'w') as f:
        json.dump(config, f, indent=2)


def randomize_lora_weights(peft_model, per_layer_noise_scales, target_layers, init_scale=1.0):
    """
    Randomize LoRA weights in-place with per-layer noise scaling.

    Args:
        peft_model: PEFT model with LoRA
        per_layer_noise_scales: Dict mapping layer_num -> base noise scale
        target_layers: List of layer numbers to randomize
        init_scale: Multiplier applied to per_layer_noise_scales (default 1.0)
    """
    for name, param in peft_model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            # Extract layer number from name
            layer_num = None
            for i in target_layers:
                if f'layers.{i}.' in name:
                    layer_num = i
                    break

            if layer_num is not None and layer_num in per_layer_noise_scales:
                scale = per_layer_noise_scales[layer_num] * init_scale
                with torch.no_grad():
                    param.normal_(mean=0.0, std=scale)


def create_initial_population(model_name, output_dir, population_size, per_layer_noise_scales, lora_config, init_scale=1.0):
    """
    Create entire initial population by loading model once and randomizing weights.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save LoRAs
        population_size: Number of LoRAs to create
        per_layer_noise_scales: Dict mapping layer_num -> base noise scale
        lora_config: LoRA configuration dict
        init_scale: Multiplier for initialization noise (default 1.0)
    """
    rank = lora_config['rank']
    alpha = lora_config['alpha']
    target_layers = lora_config['target_layers']
    target_modules = lora_config['target_modules']

    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print("  ✓ Base model loaded")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        layers_to_transform=target_layers,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )

    peft_model = get_peft_model(model, config)

    lora_paths = []
    for i in range(population_size):
        lora_path = output_dir / f"gen0_lora_{i}"
        if not lora_path.exists():
            randomize_lora_weights(peft_model, per_layer_noise_scales, target_layers, init_scale)
            peft_model.save_pretrained(lora_path)
            if (i + 1) % 10 == 0:
                print(f"  Created {i + 1}/{population_size} LoRAs...")
        lora_paths.append(lora_path)

    del model, peft_model
    torch.cuda.empty_cache()

    return lora_paths


# ============================================================================
# ES Algorithm
# ============================================================================

def compute_es_update(lora_paths, rewards, base_lora_path, learning_rate):
    """Compute ES update: weighted average based on normalized rewards."""
    rewards_array = np.array(rewards)
    rewards_mean = rewards_array.mean()
    rewards_std = rewards_array.std()

    if rewards_std < 1e-8:
        rewards_normalized = np.zeros_like(rewards_array)
    else:
        rewards_normalized = (rewards_array - rewards_mean) / rewards_std

    if base_lora_path is not None:
        base_config, base_weights = load_lora_weights(base_lora_path)
    else:
        base_config, base_weights = load_lora_weights(lora_paths[0])
        base_weights = {k: torch.zeros_like(v) for k, v in base_weights.items()}

    update = {key: torch.zeros_like(base_weights[key]) for key in base_weights.keys()}

    for i, lora_path in enumerate(lora_paths):
        _, weights = load_lora_weights(lora_path)
        weight = rewards_normalized[i]

        for key in update.keys():
            if key in weights:
                update[key] += weight * weights[key]

    N = len(lora_paths)
    for key in update.keys():
        update[key] = update[key] / N * learning_rate

    new_weights = {key: base_weights[key] + update[key] for key in base_weights.keys()}

    return base_config, new_weights


def perturb_lora(config, weights, per_layer_noise_scales, seed, perturb_scale=1.0):
    """
    Perturb LoRA weights with Gaussian noise.

    Args:
        config: LoRA config dict
        weights: LoRA weights dict
        per_layer_noise_scales: Dict mapping layer_num -> base noise scale
        seed: Random seed for reproducibility
        perturb_scale: Multiplier applied to per_layer_noise_scales (default 1.0)

    Returns:
        config, perturbed_weights
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    perturbed_weights = {}
    for key, weight in weights.items():
        if "layers." in key:
            layer_num = int(key.split("layers.")[1].split(".")[0])
            if layer_num in per_layer_noise_scales:
                noise_scale = per_layer_noise_scales[layer_num] * perturb_scale
            else:
                raise ValueError(f"No noise scale found for layer {layer_num}")
        else:
            raise ValueError(f"Unexpected weight key without layer number: {key}")

        noise = torch.randn_like(weight) * noise_scale
        perturbed_weights[key] = weight + noise

    return config, perturbed_weights


# ============================================================================
# Visualization
# ============================================================================

def plot_evolution_progress(evolution_history, output_dir, generation_num):
    """Plot evolution progress with 20th/80th percentile bands."""
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generations = [h['generation'] for h in evolution_history]
    mean_rewards = [h['mean_reward'] for h in evolution_history]
    best_rewards = [max(h['rewards']) for h in evolution_history]

    # Compute 20th and 80th percentiles
    p20_rewards = [np.percentile(h['rewards'], 20) for h in evolution_history]
    p80_rewards = [np.percentile(h['rewards'], 80) for h in evolution_history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_rewards, 'b-o', label='Mean Reward', linewidth=2)
    plt.fill_between(generations,
                     p20_rewards,
                     p80_rewards,
                     alpha=0.3, color='blue', label='20th-80th percentile')
    plt.plot(generations, best_rewards, 'g--^', label='Best Reward', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('ES Evolution Progress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "evolution.png", dpi=150)
    plt.close()


def plot_time_breakdown(time_tracker, output_dir):
    """Plot time breakdown as a horizontal bar chart."""
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate time by category
    categories = {}
    for key, duration in time_tracker.items():
        if key.startswith('gen'):
            # Generation-specific timings (e.g., "gen0_loading", "gen1_es_update")
            # Remove generation prefix to get category
            parts = key.split('_', 1)  # Split into ['genN', 'rest']
            if len(parts) >= 2:
                category = parts[1]  # "loading", "evaluation", "es_update", "create_loras", etc.
                if category not in categories:
                    categories[category] = 0.0
                categories[category] += duration
        else:
            # One-time setup timings
            categories[key] = duration

    # Sort by total time
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    cat_names = [c[0] for c in sorted_categories]
    cat_times = [c[1] for c in sorted_categories]

    # Determine colors: per-generation operations vs one-time setup
    per_gen_categories = {'evaluation', 'loading', 'unloading', 'es_update', 'create_loras'}
    colors = ['coral' if name in per_gen_categories else 'steelblue' for name in cat_names]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(cat_names))
    bars = ax.barh(y_pos, cat_times, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_names, fontsize=11)
    ax.set_xlabel('Total Time (seconds)', fontsize=12)
    ax.set_title('Time Breakdown by Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='coral', label='Per-generation'),
        Patch(facecolor='steelblue', label='One-time setup')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add time labels on bars
    for i, (name, time_val) in enumerate(zip(cat_names, cat_times)):
        ax.text(time_val, i, f'  {time_val:.1f}s', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "time_breakdown.png", dpi=150)
    plt.close()

    # Print summary
    total_time = sum(cat_times)
    print("\n" + "=" * 80)
    print("TIME BREAKDOWN SUMMARY")
    print("=" * 80)
    for name, time_val in sorted_categories:
        percentage = (time_val / total_time) * 100 if total_time > 0 else 0
        print(f"  {name:30s}: {time_val:8.2f}s ({percentage:5.1f}%)")
    print(f"  {'TOTAL':30s}: {total_time:8.2f}s")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def main(config_path, overwrite=False, debug=False):
    """Main ES evolution loop."""

    # Initialize time tracker
    time_tracker = {}
    experiment_start = time.time()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config - FAIL FAST
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")
    if 'task' not in config:
        raise ValueError("FATAL: 'task' section required in config")
    if 'model' not in config:
        raise ValueError("FATAL: 'model' section required in config")
    if 'es' not in config:
        raise ValueError("FATAL: 'es' section required in config")

    # Initialize output directory
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Create subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'loras').mkdir(parents=True, exist_ok=True)

    # Copy config to output_dir
    import shutil
    shutil.copy(config_path, output_dir / "config.yaml")

    # Extract config values
    seed = config.get('seed', 42)
    model_name = config['model']['name']
    task_name = config['task']['name']
    train_data = Path(config['task']['train_data'])
    test_data = Path(config['task'].get('test_data', config['task']['train_data']))

    num_generations = config['es']['num_generations']
    population_size = config['es']['population_size']
    learning_rate = config['es']['learning_rate']
    init_scale = config['es']['init_scale']
    perturb_scale = config['es']['perturb_scale']

    max_tokens = config['evaluation']['max_tokens']
    samples_per_gen = config['evaluation'].get('samples_per_generation', None)
    test_samples = config['evaluation'].get('test_samples', None)

    # Determine parallel workers
    import os
    n_cpus = os.cpu_count() or 1
    max_workers = min(population_size, n_cpus)

    # Initialize task and load data
    task = get_task(task_name, config['task'])
    task.load_data(seed=seed)
    train_examples_full = task.train_data
    test_examples_full = task.test_data

    # Setup training sample cycling
    if samples_per_gen is None or samples_per_gen >= len(train_examples_full):
        # Use all samples every generation (conciseness mode)
        use_cycling = False
        samples_per_gen = len(train_examples_full)
    else:
        # Cycle through subsets (countdown mode)
        use_cycling = True

    # Setup test sample subset (fixed, not cycling)
    if test_samples is None or test_samples >= len(test_examples_full):
        test_examples = test_examples_full
    else:
        # Use fixed subset from beginning
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(test_examples_full))[:test_samples]
        test_examples = [test_examples_full[i] for i in sorted(indices)]

    print("=" * 80)
    print("ES EVOLUTION WITH PARALLEL MULTI-LORA BATCHING")
    print("=" * 80)
    print(f"  Model: {model_name}")
    print(f"  Task: {task_name}")
    print(f"  Num generations: {num_generations}")
    print(f"  Population size: {population_size}")
    print(f"  Learning rate (α): {learning_rate}")
    print(f"  Init scale: {init_scale}")
    print(f"  Perturb scale: {perturb_scale}")
    print(f"  Training examples (total): {len(train_examples_full)}")
    print(f"  Training examples (per gen): {samples_per_gen}")
    if use_cycling:
        print(f"  Training mode: Cycling subsets")
    else:
        print(f"  Training mode: Full dataset each generation")
    print(f"  Test examples (total): {len(test_examples_full)}")
    print(f"  Test examples (used): {len(test_examples)}")
    if len(test_examples) < len(test_examples_full):
        print(f"  Test mode: Fixed random subset")
    else:
        print(f"  Test mode: Full test set")
    print(f"  Output dir: {output_dir}")
    print(f"  Parallel workers: {max_workers}")
    print()

    # Step 1: Compute per-layer noise scales from base model
    print("Step 1: Analyzing base model weight magnitudes...")
    t_start = time.time()
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    per_layer_noise_scales = {}
    target_layers = config['lora']['target_layers']
    target_modules = config['lora']['target_modules']

    # Compute base noise scale as 10% of mean weight magnitude per layer
    for layer_num in target_layers:
        layer_magnitudes = []
        for name, param in base_model.named_parameters():
            if f"layers.{layer_num}." in name and any(t in name for t in target_modules):
                layer_magnitudes.append(param.abs().mean().item())

        if layer_magnitudes:
            mean_magnitude = np.mean(layer_magnitudes)
            # Base noise is 10% of mean magnitude (will be multiplied by init_scale/perturb_scale later)
            per_layer_noise_scales[layer_num] = mean_magnitude * 0.1

    del base_model
    torch.cuda.empty_cache()
    time_tracker['weight_analysis'] = time.time() - t_start

    print(f"✓ Computed base noise scales (10% of mean magnitude) for {len(per_layer_noise_scales)} layers")
    for layer_num in sorted(per_layer_noise_scales.keys())[:3]:
        print(f"    Layer {layer_num}: {per_layer_noise_scales[layer_num]:.6f}")
    print()

    # Step 2: Create initial population
    print("Step 2: Creating initial population...")
    print(f"  Using init_scale={init_scale} (multiplier on per-layer base noise)")
    t_start = time.time()
    loras_dir = output_dir / "loras"
    initial_loras = create_initial_population(
        model_name, loras_dir, population_size, per_layer_noise_scales, config['lora'], init_scale
    )
    time_tracker['init_population'] = time.time() - t_start
    print(f"✓ Created {population_size} initial LoRAs")
    print()

    # Step 3: Launch SGLang server
    print("Step 3: Launching SGLang server...")
    t_start = time.time()
    server_process, port = launch_server_cmd(f"""
        python3 -m sglang.launch_server \
            --model-path {model_name} \
            --lora-paths dummy={initial_loras[0]} \
            --max-loras-per-batch {population_size} \
            --lora-backend triton \
            --disable-radix-cache \
            --port 0
    """)
    print(f"✓ Server launched on port {port}")

    evaluator = SGLangLoRAEvaluator(
        model_name=config['model']['name'],
        temperature=config['evaluation']['temperature']
    )
    evaluator.set_port(port)

    wait_for_server(f"http://localhost:{port}")
    time_tracker['server_launch'] = time.time() - t_start
    print("✓ Server ready")
    print()

    evolution_history = []

    try:
        # Unload dummy LoRA
        evaluator.unload_lora("dummy")

        # Generation 0
        print("=" * 80)
        print("GENERATION 0: Evaluating initial population")
        print("=" * 80)

        # Select training subset for this generation
        if use_cycling:
            start_idx = 0
            end_idx = min(samples_per_gen, len(train_examples_full))
            train_examples = train_examples_full[start_idx:end_idx]
            print(f"  Using samples {start_idx}-{end_idx-1} ({len(train_examples)} examples)")
        else:
            train_examples = train_examples_full

        t_load = time.time()
        print(f"\nLoading {population_size} LoRAs...")
        for i, lora_path in enumerate(initial_loras):
            lora_name = f"gen0_lora{i}"
            evaluator.load_lora(lora_name, lora_path)
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{population_size}...")
        print(f"✓ All {population_size} LoRAs loaded")
        time_tracker['gen0_loading'] = time.time() - t_load

        t_eval = time.time()
        lora_names = [f"gen0_lora{i}" for i in range(population_size)]
        gen0_rewards, gen0_results = evaluate_all_loras_parallel(
            evaluator, lora_names, train_examples, task, max_tokens, max_workers
        )
        time_tracker['gen0_evaluation'] = time.time() - t_eval

        evolution_history.append({
            'generation': 0,
            'lora_paths': [str(p) for p in initial_loras],
            'rewards': gen0_rewards,
            'mean_reward': np.mean(gen0_rewards),
            'std_reward': np.std(gen0_rewards),
            'results': gen0_results,
        })

        print(f"\nGeneration 0 summary:")
        print(f"  Mean reward: {np.mean(gen0_rewards):.4f}")
        print(f"  Std reward: {np.std(gen0_rewards):.4f}")
        print(f"  Best reward: {np.max(gen0_rewards):.4f}")

        plot_evolution_progress(evolution_history, output_dir, 0)

        # Evolution loop
        for gen in range(1, num_generations):
            print("\n" + "=" * 80)
            print(f"GENERATION {gen}")
            print("=" * 80)

            # Select training subset for this generation (cycling)
            if use_cycling:
                start_idx = (gen * samples_per_gen) % len(train_examples_full)
                end_idx = start_idx + samples_per_gen
                if end_idx <= len(train_examples_full):
                    train_examples = train_examples_full[start_idx:end_idx]
                else:
                    # Wrap around to beginning
                    train_examples = train_examples_full[start_idx:] + train_examples_full[:end_idx - len(train_examples_full)]
                print(f"  Using samples {start_idx}-{(start_idx + len(train_examples) - 1) % len(train_examples_full)} ({len(train_examples)} examples)")
            else:
                train_examples = train_examples_full

            prev_loras = [Path(p) for p in evolution_history[-1]['lora_paths']]
            prev_rewards = evolution_history[-1]['rewards']

            # ES update
            print(f"\nStep 1: Computing ES update (α={learning_rate})...")
            t_update = time.time()
            prev_base_path = loras_dir / f"gen{gen-1}_base" if gen > 1 else None
            base_config, base_weights = compute_es_update(
                prev_loras, prev_rewards, prev_base_path, learning_rate
            )

            base_lora_path = loras_dir / f"gen{gen}_base"
            save_lora_weights(base_lora_path, base_config, base_weights)
            time_tracker[f'gen{gen}_es_update'] = time.time() - t_update
            print(f"  ✓ Saved base LoRA")

            # Create new generation
            print(f"\nStep 2: Creating {population_size} perturbed LoRAs...")
            print(f"  Using perturb_scale={perturb_scale} (multiplier on per-layer base noise)")
            t_create = time.time()
            new_lora_paths = []
            for i in range(population_size):
                perturbed_config, perturbed_weights = perturb_lora(
                    base_config, base_weights, per_layer_noise_scales, gen * 1000 + i, perturb_scale
                )
                perturbed_path = loras_dir / f"gen{gen}_lora_{i}"
                save_lora_weights(perturbed_path, perturbed_config, perturbed_weights)
                new_lora_paths.append(perturbed_path)
            time_tracker[f'gen{gen}_create_loras'] = time.time() - t_create
            print(f"  ✓ Created {population_size} new LoRAs")

            # Unload old generation
            print(f"\nStep 3: Unloading generation {gen-1}...")
            t_unload = time.time()
            for i in range(population_size):
                old_lora_name = f"gen{gen-1}_lora{i}"
                evaluator.unload_lora(old_lora_name)
            time_tracker[f'gen{gen}_unloading'] = time.time() - t_unload
            print(f"  ✓ Unloaded {population_size} old LoRAs")

            # Load new generation
            print(f"\nStep 4: Loading generation {gen}...")
            t_load = time.time()
            for i, lora_path in enumerate(new_lora_paths):
                lora_name = f"gen{gen}_lora{i}"
                evaluator.load_lora(lora_name, lora_path)
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{population_size}...")
            time_tracker[f'gen{gen}_loading'] = time.time() - t_load
            print(f"✓ All {population_size} new LoRAs loaded")

            # Evaluate
            print(f"\nStep 5: Evaluating generation {gen}...")
            t_eval = time.time()
            new_lora_names = [f"gen{gen}_lora{i}" for i in range(population_size)]
            new_rewards, new_results = evaluate_all_loras_parallel(
                evaluator, new_lora_names, train_examples, task, max_tokens, max_workers
            )
            time_tracker[f'gen{gen}_evaluation'] = time.time() - t_eval

            evolution_history.append({
                'generation': gen,
                'lora_paths': [str(p) for p in new_lora_paths],
                'rewards': new_rewards,
                'mean_reward': np.mean(new_rewards),
                'std_reward': np.std(new_rewards),
                'results': new_results,
            })

            print(f"\nGeneration {gen} summary:")
            print(f"  Mean reward: {np.mean(new_rewards):.4f}")
            print(f"  Std reward: {np.std(new_rewards):.4f}")
            print(f"  Best reward: {np.max(new_rewards):.4f}")

            plot_evolution_progress(evolution_history, output_dir, gen)

        # Final evaluation
        print("\n" + "=" * 80)
        print("FINAL EVALUATION: Best LoRA on test set")
        print("=" * 80)

        final_gen = evolution_history[-1]
        best_idx = np.argmax(final_gen['rewards'])
        best_lora_name = f"gen{num_generations-1}_lora{best_idx}"

        print(f"Best LoRA: {best_lora_name}")
        print(f"Training reward: {final_gen['rewards'][best_idx]:.4f}")
        print(f"\nEvaluating on test set...")

        test_reward, test_results = evaluate_lora_on_task(
            evaluator, best_lora_name, test_examples, task, max_tokens
        )

        print(f"\nTest set results:")
        print(f"  Mean reward: {test_reward:.4f}")

        # Record total experiment time
        time_tracker['total_experiment'] = time.time() - experiment_start

        # Save results
        results_file = output_dir / "results" / "evolution_history.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(evolution_history, f, indent=2)
        print(f"\n✓ Saved evolution history to: {results_file}")

        # Save time tracker
        time_file = output_dir / "results" / "time_breakdown.json"
        with open(time_file, 'w') as f:
            json.dump(time_tracker, f, indent=2)
        print(f"✓ Saved time breakdown to: {time_file}")

        # Plot time breakdown
        plot_time_breakdown(time_tracker, output_dir)
        print(f"✓ Saved time breakdown plot to: {output_dir / 'figures' / 'time_breakdown.png'}")

    finally:
        print("\n" + "=" * 80)
        print("Cleaning up...")
        print("=" * 80)
        terminate_process(server_process)
        print("✓ Server terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
