"""Conciseness task: train model to generate responses with target length."""

import json


class ConcisenessTask:
    """Task for training conciseness - matching target text length."""

    def __init__(self, config):
        """
        Initialize conciseness task.

        Args:
            config: Task config dict with 'train_data' and 'test_data' paths
        """
        self.config = config
        self.train_data = None
        self.test_data = None

    def load_data(self, seed=None):
        """Load train and test data."""
        with open(self.config['train_data'], 'r') as f:
            self.train_data = json.load(f)
        with open(self.config['test_data'], 'r') as f:
            self.test_data = json.load(f)

    def compute_reward(self, generated_text, example):
        """
        Compute reward for conciseness task.

        Reward = -|len(generated) - len(target)|

        Args:
            generated_text: Generated text from model
            example: Dict containing 'target' key

        Returns:
            Float reward value
        """
        target_text = example['target']
        return -abs(len(generated_text) - len(target_text))
