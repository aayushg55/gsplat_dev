#!/bin/bash

# Run tests without PyTorch
echo "Running determinism tests with default backend..."
pytest test_training_determinism.py -s

# Run tests with PyTorch
echo "Running determinism tests with PyTorch backend..."
pytest test_training_determinism.py -s --use_pytorch

