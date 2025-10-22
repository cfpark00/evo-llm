"""Countdown task: train model to solve arithmetic problems with given numbers."""

import json
import re
import numpy as np


class CountdownTask:
    """Task for training arithmetic problem solving with format constraints."""

    def __init__(self, config):
        """
        Initialize countdown task.

        Args:
            config: Task config dict with 'train_data' and 'test_data' paths
        """
        self.config = config
        self.train_data = None
        self.test_data = None

    def load_data(self, seed=None):
        """
        Load and split data into train/test sets.

        For countdown, we load the full dataset and split it.
        The split is deterministic based on seed.
        """
        # Load full dataset
        with open(self.config['train_data'], 'r') as f:
            full_data = json.load(f)

        # Set seed for reproducible split
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(42)

        # Shuffle and split 80/20
        indices = rng.permutation(len(full_data))
        split_idx = int(0.8 * len(full_data))

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Normalize data format: add 'prompt' key from 'context'
        self.train_data = []
        for i in train_indices:
            example = full_data[i].copy()
            example['prompt'] = example.get('context', '')
            self.train_data.append(example)

        self.test_data = []
        for i in test_indices:
            example = full_data[i].copy()
            example['prompt'] = example.get('context', '')
            self.test_data.append(example)

    def _format_reward(self, response, end_token=None):
        """
        Check if response follows format <think>...</think><answer>...</answer>
        """
        # Strip end token if present
        if end_token and response.endswith(end_token):
            response = response[: -len(end_token)]

        think_regex = r"<think>.*?<\/think>"
        answer_regex = r"<answer>.*?<\/answer>"
        full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

        think_match = re.search(think_regex, response, re.DOTALL)
        answer_match = re.search(answer_regex, response, re.DOTALL)
        full_format_match = re.match(full_format_regex, response, re.DOTALL)

        if full_format_match:
            return 1.0

        reward = 0.0

        if think_match:
            reward += 0.1

        if answer_match:
            reward += 0.5

        return reward

    def _answer_reward(self, response, numbers, target):
        """
        Check if last <answer>...</answer> uses all numbers exactly once and evaluates to target.
        """
        answer_regex = r"<answer>(.*?)<\/answer>"
        all_matches = re.findall(answer_regex, response, re.DOTALL)

        if not all_matches:
            return 0.0

        # Only check the last answer
        answer_content = all_matches[-1]

        allowed_chars = r"^[0-9+\-*/() ]+$"

        if not answer_content:
            return 0.0
        if not re.match(allowed_chars, answer_content):
            return 0.0

        # Check numbers used
        used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_numbers) != sorted(numbers):
            return 0.0

        # Try evaluating
        try:
            result = eval(answer_content, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0
        except:
            return 0.0

        return 0.0

    def compute_reward(self, generated_text, example):
        """
        Compute reward for countdown task.

        Total reward = 0.1 * format_reward + answer_reward

        Args:
            generated_text: Generated text from model
            example: Dict with 'numbers' and 'target' keys

        Returns:
            Float reward value
        """
        numbers = example.get('numbers', [])
        target = example.get('target')

        format_reward = self._format_reward("<think>" + generated_text, end_token=None)
        answer_reward = self._answer_reward(generated_text, numbers, target)

        return format_reward * 0.1 + answer_reward
