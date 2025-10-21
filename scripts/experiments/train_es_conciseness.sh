#!/bin/bash
# Run ES evolution experiment on conciseness task

uv run python src/scripts/es_train.py configs/experiments/es_conciseness.yaml "$@"
