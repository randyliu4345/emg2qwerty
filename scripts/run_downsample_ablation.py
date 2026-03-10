#!/usr/bin/env python3
"""Run downsampling factor ablation experiments with different temporal downsampling rates.

This script runs training experiments with different downsample_factor values:
- 1 (no downsampling - baseline, original sampling rate)
- 2 (2x downsampling - half the temporal resolution)
- 4 (4x downsampling - quarter the temporal resolution)
- 8 (8x downsampling - eighth the temporal resolution)

Each experiment runs for 15 epochs, then runs validation to check validation accuracy.

Note: downsample_factor controls temporal downsampling of the EMG signal.
A factor of N means every Nth sample is kept, reducing temporal resolution.
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


def run_experiment(downsample_factor: int, epochs: int = 15):
    """Run a single experiment with specified downsampling factor.
    
    Args:
        downsample_factor: Temporal downsampling factor (1 = no downsampling)
        epochs: Number of epochs to train
        
    Returns:
        Tuple of (success: bool, checkpoint_path: Path | None)
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: downsample_factor={downsample_factor}")
    if downsample_factor == 1:
        print("No downsampling (baseline)")
    else:
        print(f"{downsample_factor}x downsampling ({100/downsample_factor:.1f}% of temporal resolution)")
    print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")

    hydra_run_dir = f"logs/downsample_ablation/{downsample_factor}_downsample"
    
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
        f"data.downsample_factor={downsample_factor}",
    ]
    
    # Run training
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        print(f"\n✓ Training completed: downsample_factor={downsample_factor}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed: downsample_factor={downsample_factor}")
        print(f"Error: {e}\n")
        return False, None
    
    # Find the checkpoint
    checkpoint_path = find_most_recent_checkpoint()
    if checkpoint_path is None:
        print(f"\n⚠ Could not find checkpoint after training: downsample_factor={downsample_factor}")
        return False, None
    
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Run validation and testing
    print(f"\n{'='*80}")
    print(f"Running validation and testing: downsample_factor={downsample_factor}")
    print(f"{'='*80}\n")
    
    val_test_cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        f"checkpoint={checkpoint_path}",
        "train=False",
        "trainer.accelerator=gpu",
        "decoder=ctc_greedy",
        f"data.downsample_factor={downsample_factor}",
    ]
    
    try:
        result = subprocess.run(val_test_cmd, check=True, capture_output=False)
        print(f"\n✓ Validation and testing completed: downsample_factor={downsample_factor}\n")
        return True, checkpoint_path
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Validation and testing failed: downsample_factor={downsample_factor}")
        print(f"Error: {e}\n")
        return False, checkpoint_path


def main():
    """Run all downsampling factor ablation experiments."""
    # Define downsampling factor configurations
    experiments = [
        (1, 50),  # No downsampling (baseline)
        (2, 50),  # 2x downsampling
        (4, 50),  # 4x downsampling
        (8, 50),  # 8x downsampling
    ]
    
    print("="*80)
    print("Downsampling Factor Ablation Experiments")
    print("="*80)
    print(f"Running {len(experiments)} experiments with different downsampling factors")
    print("After training, validation and test accuracy will be checked")
    print("CER will be logged to stdout/stderr each epoch")
    print("="*80)
    
    results = []
    for downsample_factor, epochs in experiments:
        success, checkpoint_path = run_experiment(downsample_factor, epochs)
        results.append((downsample_factor, success, checkpoint_path))
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    for downsample_factor, success, checkpoint_path in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        checkpoint_info = f" (checkpoint: {checkpoint_path})" if checkpoint_path else ""
        if downsample_factor == 1:
            print(f"downsample_factor={downsample_factor} (no downsampling): {status}{checkpoint_info}")
        else:
            print(f"downsample_factor={downsample_factor} ({downsample_factor}x downsampling): {status}{checkpoint_info}")
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
