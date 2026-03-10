#!/usr/bin/env python3
"""Run training fraction ablation experiments with different amounts of training data.

This script runs training experiments with different train_fraction values:
- 1.0 (100% of training data - baseline)
- 0.5 (50% of training data)
- 0.25 (25% of training data)
- 0.1 (10% of training data)

Each experiment runs for 15 epochs, then runs validation to check validation accuracy.
"""

import subprocess
import sys
from pathlib import Path


def find_most_recent_checkpoint(logs_dir: Path = Path("logs")) -> Path | None:
    """Find the most recent checkpoint in the logs directory.
    
    Args:
        logs_dir: Path to the logs directory
        
    Returns:
        Path to the most recent checkpoint, or None if not found
    """
    if not logs_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoints = []
    for date_dir in logs_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            checkpoint_file = time_dir / "checkpoints" / "last.ckpt"
            if checkpoint_file.exists():
                checkpoints.append((checkpoint_file.stat().st_mtime, checkpoint_file))
    
    if not checkpoints:
        return None
    
    # Return the most recent checkpoint
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def run_experiment(train_fraction: float, epochs: int = 15):
    """Run a single experiment with specified training fraction.
    
    Args:
        train_fraction: Fraction of training examples to use (0.0 to 1.0)
        epochs: Number of epochs to train
        
    Returns:
        Tuple of (success: bool, checkpoint_path: Path | None)
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: train_fraction={train_fraction}")
    print(f"Using {train_fraction*100:.0f}% of training data")
    print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")

    hydra_run_dir = f"logs/train_fraction_ablation/{train_fraction}_fraction"
    
    # Build training command
    train_cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        'user=single_user',
        "model=lstm_ctc",
        "module.lstm_layers=2",
        "module.lstm_hidden_size=128",
        "module.bidirectional=true",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        f"hydra.run.dir={hydra_run_dir}",
        "lr_scheduler.scheduler.max_epochs=150",
        f"trainer.max_epochs={epochs}",
        f"data.train_fraction={train_fraction}",
    ]
    
    # Run training
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        print(f"\n✓ Training completed: train_fraction={train_fraction}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed: train_fraction={train_fraction}")
        print(f"Error: {e}\n")
        return False, None
    
    # Find the checkpoint
    checkpoint_path = find_most_recent_checkpoint()
    if checkpoint_path is None:
        print(f"\n⚠ Could not find checkpoint after training: train_fraction={train_fraction}")
        return False, None
    
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Run validation and testing
    print(f"\n{'='*80}")
    print(f"Running validation and testing: train_fraction={train_fraction}")
    print(f"{'='*80}\n")
    
    val_test_cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        f"checkpoint={checkpoint_path}",
        "train=False",
        "trainer.accelerator=gpu",
        "decoder=ctc_greedy",
        f"data.train_fraction={train_fraction}",
    ]
    
    try:
        result = subprocess.run(val_test_cmd, check=True, capture_output=False)
        print(f"\n✓ Validation and testing completed: train_fraction={train_fraction}\n")
        return True, checkpoint_path
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Validation and testing failed: train_fraction={train_fraction}")
        print(f"Error: {e}\n")
        return False, checkpoint_path


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
    print("After training, validation and test accuracy will be checked")
    print("CER will be logged to stdout/stderr each epoch")
    print("="*80)
    
    results = []
    for train_fraction, epochs in experiments:
        success, checkpoint_path = run_experiment(train_fraction, epochs)
        results.append((train_fraction, success, checkpoint_path))
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    for train_fraction, success, checkpoint_path in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        checkpoint_info = f" (checkpoint: {checkpoint_path})" if checkpoint_path else ""
        print(f"train_fraction={train_fraction:.2f} ({train_fraction*100:.0f}%): {status}{checkpoint_info}")
    print("="*80)
    
    # Check if all succeeded
    all_success = all(success for _, success, _ in results)
    if not all_success:
        print("\n⚠ Some experiments failed. Check the logs above.")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
