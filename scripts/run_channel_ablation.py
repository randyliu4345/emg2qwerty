#!/usr/bin/env python3
"""Run channel ablation experiments with different numbers of channels.

This script runs training experiments with different channel configurations:
- 32 channels (all channels, 16 per band)
- 16 channels (8 per band)
- 8 channels (4 per band)
- 4 channels (2 per band)

Each experiment runs for 10-20 epochs and logs CER for each epoch.
"""

import subprocess
import sys
from pathlib import Path


def run_experiment(num_channels: int, channel_indices: list[int] | None, epochs: int = 15):
    """Run a single experiment with specified channel configuration.
    
    Args:
        num_channels: Total number of channels (for logging/identification)
        channel_indices: List of channel indices to use, or None for all channels
        epochs: Number of epochs to train
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: {num_channels} channels")
    if channel_indices is not None:
        print(f"Channel indices: {channel_indices}")
    else:
        print("Using all channels")
    print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        f"trainer.max_epochs={epochs}",
    ]
    
    # Add channel_indices if specified
    # Hydra accepts lists as [0,1,2,3] format
    if channel_indices is not None:
        # Format as Hydra list: [0,1,2,3] (no spaces)
        indices_str = "[" + ",".join(map(str, channel_indices)) + "]"
        cmd.append(f"data.channel_indices={indices_str}")
    else:
        # Use null to indicate all channels
        cmd.append("data.channel_indices=null")
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Experiment completed: {num_channels} channels\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed: {num_channels} channels")
        print(f"Error: {e}\n")
        return False


def main():
    """Run all channel ablation experiments."""
    # Define channel configurations
    # Each band has 16 channels, so:
    # - 32 total = 16 per band (use all: null or [0..15])
    # - 16 total = 8 per band (use [0..7])
    # - 8 total = 4 per band (use [0..3])
    # - 4 total = 2 per band (use [0..1])
    
    experiments = [
        (32, None, 15),  # All channels (null = use all)
        (16, list(range(8)), 15),  # 8 channels per band
        (8, list(range(4)), 15),  # 4 channels per band
        (4, list(range(2)), 15),  # 2 channels per band
    ]
    
    print("="*80)
    print("Channel Ablation Experiments")
    print("="*80)
    print(f"Running {len(experiments)} experiments with different channel configurations")
    print("Each experiment will train for 15 epochs")
    print("CER will be logged to stdout/stderr each epoch")
    print("="*80)
    
    results = []
    for num_channels, channel_indices, epochs in experiments:
        success = run_experiment(num_channels, channel_indices, epochs)
        results.append((num_channels, success))
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    for num_channels, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{num_channels:2d} channels: {status}")
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
