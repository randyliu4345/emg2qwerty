#!/usr/bin/env python3
"""Run downsampling factor ablation experiments with different temporal downsampling rates.

This script runs training experiments with different downsample_factor values:
- 1 (no downsampling - baseline, original sampling rate)
- 2 (2x downsampling - half the temporal resolution)
- 4 (4x downsampling - quarter the temporal resolution)
- 8 (8x downsampling - eighth the temporal resolution)

Each experiment runs for 15 epochs and logs CER for each epoch.

Note: downsample_factor controls temporal downsampling of the EMG signal.
A factor of N means every Nth sample is kept, reducing temporal resolution.
"""

import subprocess
import sys
from pathlib import Path


def run_experiment(downsample_factor: int, epochs: int = 15):
    """Run a single experiment with specified downsampling factor.
    
    Args:
        downsample_factor: Temporal downsampling factor (1 = no downsampling)
        epochs: Number of epochs to train
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: downsample_factor={downsample_factor}")
    if downsample_factor == 1:
        print("No downsampling (baseline)")
    else:
        print(f"{downsample_factor}x downsampling ({100/downsample_factor:.1f}% of temporal resolution)")
    print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        f"trainer.max_epochs={epochs}",
        f"data.downsample_factor={downsample_factor}",
    ]
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Experiment completed: downsample_factor={downsample_factor}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed: downsample_factor={downsample_factor}")
        print(f"Error: {e}\n")
        return False


def main():
    """Run all downsampling factor ablation experiments."""
    # Define downsampling factor configurations
    experiments = [
        (1, 15),  # No downsampling (baseline)
        (2, 15),  # 2x downsampling
        (4, 15),  # 4x downsampling
        (8, 15),  # 8x downsampling
    ]
    
    print("="*80)
    print("Downsampling Factor Ablation Experiments")
    print("="*80)
    print(f"Running {len(experiments)} experiments with different downsampling factors")
    print("Each experiment will train for 15 epochs")
    print("CER will be logged to stdout/stderr each epoch")
    print("="*80)
    
    results = []
    for downsample_factor, epochs in experiments:
        success = run_experiment(downsample_factor, epochs)
        results.append((downsample_factor, success))
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    for downsample_factor, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        if downsample_factor == 1:
            print(f"downsample_factor={downsample_factor} (no downsampling): {status}")
        else:
            print(f"downsample_factor={downsample_factor} ({downsample_factor}x downsampling): {status}")
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
