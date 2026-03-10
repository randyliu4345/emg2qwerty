#!/usr/bin/env python3
"""Run transform ablation experiments with different augmentation combinations.

This script runs training experiments with different transform configurations:
- Baseline: no gaussian noise, no time warping
- With gaussian noise only
- With time warping only
- With both gaussian noise and time warping

Each experiment runs training, then runs validation and testing.
"""

import subprocess
import sys
from pathlib import Path


def find_checkpoint_in_dir(experiment_dir: Path) -> Path | None:
    """Find the checkpoint in a specific experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Path to the checkpoint, or None if not found
    """
    checkpoint_file = experiment_dir / "checkpoints" / "last.ckpt"
    if checkpoint_file.exists():
        return checkpoint_file
    return None


def get_experiment_dir_name(use_gaussian_noise: bool, use_time_warp: bool) -> str:
    """Generate a descriptive directory name for the experiment.
    
    Args:
        use_gaussian_noise: Whether gaussian noise is used
        use_time_warp: Whether time warping is used
        
    Returns:
        Directory name for the experiment
    """
    parts = []
    if use_gaussian_noise:
        parts.append("gaussian_noise")
    if use_time_warp:
        parts.append("time_warp")
    
    if not parts:
        return "baseline"
    
    return "_".join(parts)


def run_experiment(
    name: str,
    use_gaussian_noise: bool,
    use_time_warp: bool,
    epochs: int | None = None,
):
    """Run a single experiment with specified transform configuration.
    
    Args:
        name: Name/description of the experiment
        use_gaussian_noise: Whether to include gaussian noise transform
        use_time_warp: Whether to include time warping transform
        epochs: Number of epochs to train (None uses default from config)
        
    Returns:
        Tuple of (success: bool, checkpoint_path: Path | None, experiment_dir: Path | None)
    """
    # Generate well-named directory for this experiment
    experiment_dir_name = get_experiment_dir_name(use_gaussian_noise, use_time_warp)
    # Use simple directory structure: logs/transform_ablation/experiment_name
    hydra_run_dir = f"logs/transform_ablation/{experiment_dir_name}"
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {name}")
    print(f"Gaussian noise: {use_gaussian_noise}")
    print(f"Time warping: {use_time_warp}")
    print(f"Log directory: {hydra_run_dir}")
    if epochs is not None:
        print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")
    
    # Build base train transforms list
    # Base transforms: to_tensor, band_rotation, temporal_jitter
    # Note: Using ${variable} syntax for Hydra variable interpolation
    train_transforms = [
        "${to_tensor}",
        "${band_rotation}",
        "${temporal_jitter}",
    ]
    
    # Add optional transforms before logspec (they operate on raw signals)
    if use_gaussian_noise:
        train_transforms.append("${gaussian_noise}")
    
    if use_time_warp:
        train_transforms.append("${time_warp}")
    
    # Add required transforms after optional ones
    train_transforms.extend([
        "${logspec}",
        "${specaug}",
    ])
    
    # Format as Hydra list: [item1,item2,item3] (no spaces)
    # Hydra will resolve ${variable} references from the config
    transforms_str = "[" + ",".join(train_transforms) + "]"
    
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
        f"transforms.train={transforms_str}",
        f"hydra.run.dir={hydra_run_dir}",
        "lr_scheduler.scheduler.max_epochs=150",
    ]
    
    # Add epochs override if specified
    if epochs is not None:
        train_cmd.append(f"trainer.max_epochs={epochs}")
    
    # Run training
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        print(f"\n✓ Training completed: {name}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed: {name}")
        print(f"Error: {e}\n")
        return False, None, None
    
    # Find the checkpoint in the experiment directory
    experiment_dir = Path(hydra_run_dir)
    checkpoint_path = find_checkpoint_in_dir(experiment_dir)
    if checkpoint_path is None:
        print(f"\n⚠ Could not find checkpoint after training: {name}")
        print(f"Expected checkpoint in: {experiment_dir / 'checkpoints' / 'last.ckpt'}")
        return False, None, experiment_dir
    
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Run validation and testing
    print(f"\n{'='*80}")
    print(f"Running validation and testing: {name}")
    print(f"{'='*80}\n")
    
    val_test_cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        'user=single_user',
        "model=lstm_ctc",
        "module.lstm_layers=2",
        "module.lstm_hidden_size=128",
        "module.bidirectional=true",
        f"checkpoint={checkpoint_path}",
        "train=False",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "decoder=ctc_greedy",
        f"hydra.run.dir={hydra_run_dir}",
    ]
    
    try:
        result = subprocess.run(val_test_cmd, check=True, capture_output=False)
        print(f"\n✓ Validation and testing completed: {name}\n")
        return True, checkpoint_path, experiment_dir
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Validation and testing failed: {name}")
        print(f"Error: {e}\n")
        return False, checkpoint_path, experiment_dir


def main():
    """Run all transform ablation experiments."""
    # Define experiments: (name, use_gaussian_noise, use_time_warp)
    experiments = [
        ("Baseline (no gaussian noise, no time warping)", False, False),
        ("With gaussian noise only", True, False),
        ("With time warping only", False, True),
        ("With both gaussian noise and time warping", True, True),
    ]
    
    print("="*80)
    print("Transform Ablation Experiments")
    print("="*80)
    print(f"Running {len(experiments)} experiments with different transform configurations")
    print("Each experiment will train for 50 epochs, then run validation and test")
    print("CER will be logged to stdout/stderr each epoch")
    print("="*80)
    
    results = []
    for name, use_gaussian_noise, use_time_warp in experiments:
        success, checkpoint_path, experiment_dir = run_experiment(
            name, use_gaussian_noise, use_time_warp, epochs=50
        )
        results.append((name, success, checkpoint_path, experiment_dir))
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    for name, success, checkpoint_path, experiment_dir in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        checkpoint_info = f" (checkpoint: {checkpoint_path})" if checkpoint_path else ""
        dir_info = f" (logs: {experiment_dir})" if experiment_dir else ""
        print(f"{name}: {status}{checkpoint_info}{dir_info}")
    print("="*80)
    
    # Check if all succeeded
    all_success = all(success for _, success, _, _ in results)
    if not all_success:
        print("\n⚠ Some experiments failed. Check the logs above.")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")
        print("\nAll logs are organized in: logs/transform_ablation/")
        sys.exit(0)


if __name__ == "__main__":
    main()
