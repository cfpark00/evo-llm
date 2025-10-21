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
# Task-specific reward functions
# ============================================================================

def compute_conciseness_reward(generated_text, target_text):
    """Conciseness task: reward = -|len(generated) - len(target)|"""
    return -abs(len(generated_text) - len(target_text))


def evaluate_single_example(evaluator, lora_name, example, reward_fn, max_tokens):
    """Evaluate a single example for a given LoRA."""
    messages = [{"role": "user", "content": example['prompt']}]
    response = evaluator.generate(lora_name, messages, max_tokens=max_tokens)

    reward = reward_fn(response, example['target'])

    return {
        'example_id': example['id'],
        'prompt': example['prompt'],
        'target': example['target'],
        'response': response,
        'reward': reward,
        'generated_length': len(response),
        'target_length': len(example['target']),
    }


def evaluate_lora_on_task(evaluator, lora_name, examples, reward_fn, max_tokens):
    """Evaluate a single LoRA on all examples."""
    total_reward = 0
    results = []

    for example in examples:
        result = evaluate_single_example(evaluator, lora_name, example, reward_fn, max_tokens)
        total_reward += result['reward']
        results.append(result)

    mean_reward = total_reward / len(examples)
    return mean_reward, results


def evaluate_all_loras_parallel(evaluator, lora_names, examples, reward_fn, max_tokens, max_workers):
    """Evaluate all LoRAs in parallel."""
    print(f"\nEvaluating {len(lora_names)} LoRAs in parallel...")
    start_time = time.time()

    def eval_one_lora(lora_name):
        mean_reward, results = evaluate_lora_on_task(evaluator, lora_name, examples, reward_fn, max_tokens)
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


def randomize_lora_weights(peft_model, per_layer_noise_scales, target_layers):
    """Randomize LoRA weights in-place with per-layer noise scaling."""
    for name, param in peft_model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            # Extract layer number from name
            layer_num = None
            for i in target_layers:
                if f'layers.{i}.' in name:
                    layer_num = i
                    break

            if layer_num is not None and layer_num in per_layer_noise_scales:
                scale = per_layer_noise_scales[layer_num]
                with torch.no_grad():
                    param.normal_(mean=0.0, std=scale)


def create_initial_population(model_name, output_dir, population_size, per_layer_noise_scales, rank=1, alpha=2, target_layers=None):
    """Create entire initial population by loading model once and randomizing weights."""
    if target_layers is None:
        target_layers = list(range(10))

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
            randomize_lora_weights(peft_model, per_layer_noise_scales, target_layers)
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


def perturb_lora(config, weights, per_layer_noise_scales, seed):
    """Perturb LoRA weights with Gaussian noise."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    perturbed_weights = {}
    for key, weight in weights.items():
        if "layers." in key:
            layer_num = int(key.split("layers.")[1].split(".")[0])
            noise_scale = per_layer_noise_scales.get(layer_num, 0.1)
        else:
            noise_scale = 0.1

        noise = torch.randn_like(weight) * noise_scale
        perturbed_weights[key] = weight + noise

    return config, perturbed_weights


# ============================================================================
# Visualization
# ============================================================================

def plot_evolution_progress(evolution_history, output_dir, generation_num):
    """Plot evolution progress."""
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generations = [h['generation'] for h in evolution_history]
    mean_rewards = [h['mean_reward'] for h in evolution_history]
    std_rewards = [h['std_reward'] for h in evolution_history]
    best_rewards = [max(h['rewards']) for h in evolution_history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_rewards, 'b-o', label='Mean Reward', linewidth=2)
    plt.fill_between(generations,
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue')
    plt.plot(generations, best_rewards, 'g--^', label='Best Reward', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('ES Evolution Progress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "evolution.png", dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(config_path, overwrite=False, debug=False):
    """Main ES evolution loop."""

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
    model_name = config['model']['name']
    task_name = config['task']['name']
    train_data = Path(config['task']['train_data'])
    test_data = Path(config['task']['test_data'])

    num_generations = config['es']['num_generations']
    population_size = config['es']['population_size']
    learning_rate = config['es']['learning_rate']
    noise_multiplier = config['es']['noise_multiplier']

    max_tokens = config['evaluation']['max_tokens']

    # Determine parallel workers
    import os
    n_cpus = os.cpu_count() or 1
    max_workers = min(population_size, n_cpus)

    # Load task data
    with open(train_data, 'r') as f:
        train_examples = json.load(f)
    with open(test_data, 'r') as f:
        test_examples = json.load(f)

    # Select reward function based on task
    if task_name == "conciseness":
        reward_fn = compute_conciseness_reward
    else:
        raise ValueError(f"Unknown task: {task_name}")

    print("=" * 80)
    print("ES EVOLUTION WITH PARALLEL MULTI-LORA BATCHING")
    print("=" * 80)
    print(f"  Model: {model_name}")
    print(f"  Task: {task_name}")
    print(f"  Num generations: {num_generations}")
    print(f"  Population size: {population_size}")
    print(f"  Learning rate (α): {learning_rate}")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Test examples: {len(test_examples)}")
    print(f"  Output dir: {output_dir}")
    print(f"  Parallel workers: {max_workers}")
    print()

    # Step 1: Compute per-layer noise scales from base model
    print("Step 1: Analyzing base model weight magnitudes...")
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    per_layer_noise_scales = {}
    target_layers = list(range(10))  # Layers 0-9

    for layer_num in target_layers:
        layer_magnitudes = []
        for name, param in base_model.named_parameters():
            if f"layers.{layer_num}." in name and any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                layer_magnitudes.append(param.abs().mean().item())

        if layer_magnitudes:
            mean_magnitude = np.mean(layer_magnitudes)
            per_layer_noise_scales[layer_num] = mean_magnitude * 0.1 * noise_multiplier

    del base_model
    torch.cuda.empty_cache()

    print(f"✓ Computed noise scales for {len(per_layer_noise_scales)} layers")
    for layer_num in sorted(per_layer_noise_scales.keys())[:3]:
        print(f"    Layer {layer_num}: {per_layer_noise_scales[layer_num]:.6f}")
    print()

    # Step 2: Create initial population
    print("Step 2: Creating initial population...")
    loras_dir = output_dir / "loras"
    initial_loras = create_initial_population(
        model_name, loras_dir, population_size, per_layer_noise_scales, target_layers=target_layers
    )
    print(f"✓ Created {population_size} initial LoRAs")
    print()

    # Step 3: Launch SGLang server
    print("Step 3: Launching SGLang server...")
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

        print(f"\nLoading {population_size} LoRAs...")
        for i, lora_path in enumerate(initial_loras):
            lora_name = f"gen0_lora{i}"
            evaluator.load_lora(lora_name, lora_path)
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{population_size}...")
        print(f"✓ All {population_size} LoRAs loaded")

        lora_names = [f"gen0_lora{i}" for i in range(population_size)]
        gen0_rewards, gen0_results = evaluate_all_loras_parallel(
            evaluator, lora_names, train_examples, reward_fn, max_tokens, max_workers
        )

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

            prev_loras = [Path(p) for p in evolution_history[-1]['lora_paths']]
            prev_rewards = evolution_history[-1]['rewards']

            # ES update
            print(f"\nStep 1: Computing ES update (α={learning_rate})...")
            prev_base_path = loras_dir / f"gen{gen-1}_base" if gen > 1 else None
            base_config, base_weights = compute_es_update(
                prev_loras, prev_rewards, prev_base_path, learning_rate
            )

            base_lora_path = loras_dir / f"gen{gen}_base"
            save_lora_weights(base_lora_path, base_config, base_weights)
            print(f"  ✓ Saved base LoRA")

            # Create new generation
            print(f"\nStep 2: Creating {population_size} perturbed LoRAs...")
            new_lora_paths = []
            for i in range(population_size):
                perturbed_config, perturbed_weights = perturb_lora(
                    base_config, base_weights, per_layer_noise_scales, gen * 1000 + i
                )
                perturbed_path = loras_dir / f"gen{gen}_lora_{i}"
                save_lora_weights(perturbed_path, perturbed_config, perturbed_weights)
                new_lora_paths.append(perturbed_path)
            print(f"  ✓ Created {population_size} new LoRAs")

            # Unload old generation
            print(f"\nStep 3: Unloading generation {gen-1}...")
            for i in range(population_size):
                old_lora_name = f"gen{gen-1}_lora{i}"
                evaluator.unload_lora(old_lora_name)
            print(f"  ✓ Unloaded {population_size} old LoRAs")

            # Load new generation
            print(f"\nStep 4: Loading generation {gen}...")
            for i, lora_path in enumerate(new_lora_paths):
                lora_name = f"gen{gen}_lora{i}"
                evaluator.load_lora(lora_name, lora_path)
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{population_size}...")
            print(f"✓ All {population_size} new LoRAs loaded")

            # Evaluate
            print(f"\nStep 5: Evaluating generation {gen}...")
            new_lora_names = [f"gen{gen}_lora{i}" for i in range(population_size)]
            new_rewards, new_results = evaluate_all_loras_parallel(
                evaluator, new_lora_names, train_examples, reward_fn, max_tokens, max_workers
            )

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
            evaluator, best_lora_name, test_examples, reward_fn, max_tokens
        )

        print(f"\nTest set results:")
        print(f"  Mean reward: {test_reward:.4f}")

        # Save results
        results_file = output_dir / "results" / "evolution_history.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(evolution_history, f, indent=2)
        print(f"\n✓ Saved evolution history to: {results_file}")

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
