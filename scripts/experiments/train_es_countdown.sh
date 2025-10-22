#!/bin/bash
# Run ES evolution experiment on countdown task

uv run python src/scripts/es_train.py configs/experiments/es_countdown.yaml "$@"
