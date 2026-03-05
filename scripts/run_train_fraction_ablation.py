#!/usr/bin/env python3
"""Run training fraction ablation experiments with different amounts of training data.

This script runs training experiments with different train_fraction values:
- 1.0 (100% of training data - baseline)
- 0.5 (50% of training data)
- 0.25 (25% of training data)
- 0.1 (10% of training data)

Each experiment runs for 15 epochs and logs CER for each epoch.
"""

import subprocess
import sys
from pathlib import Path


def run_experiment(train_fraction: float, epochs: int = 15):
    """Run a single experiment with specified training fraction.
    
    Args:
        train_fraction: Fraction of training examples to use (0.0 to 1.0)
        epochs: Number of epochs to train
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: train_fraction={train_fraction}")
    print(f"Using {train_fraction*100:.0f}% of training data")
    print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        f"trainer.max_epochs={epochs}",
        f"data.train_fraction={train_fraction}",
    ]
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Experiment completed: train_fraction={train_fraction}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed: train_fraction={train_fraction}")
        print(f"Error: {e}\n")
        return False


def main():
    """Run all training fraction ablation experiments."""
    # Define training fraction configurations
    experiments = [
        (1.0, 15),   # 100% of training data (baseline)
        (0.5, 15),   # 50% of training data
        (0.25, 15),  # 25% of training data
        (0.1, 15),   # 10% of training data
    ]
    
    print("="*80)
    print("Training Fraction Ablation Experiments")
    print("="*80)
    print(f"Running {len(experiments)} experiments with different training fractions")
    print("Each experiment will train for 15 epochs")
    print("CER will be logged to stdout/stderr each epoch")
    print("="*80)
    
    results = []
    for train_fraction, epochs in experiments:
        success = run_experiment(train_fraction, epochs)
        results.append((train_fraction, success))
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    for train_fraction, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"train_fraction={train_fraction:.2f} ({train_fraction*100:.0f}%): {status}")
    print("="*80)
    
    # Check if all succeeded
    all_success = all(success for _, success in results)
    if not all_success:
        print("\n⚠ Some experiments failed. Check the logs above.")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
