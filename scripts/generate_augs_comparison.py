#!/usr/bin/env python3
"""
Generate a comparison plot overlaying validation CER from all runs in the augs directory.

Usage:
    python scripts/generate_augs_comparison.py
"""

import sys
from pathlib import Path

# Import the comparison plotting function
sys.path.insert(0, str(Path(__file__).parent))
from view_cer_history import plot_comparison


def find_runs_with_lightning_logs(base_dir: Path):
    """Find all directories that contain lightning_logs."""
    runs = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.endswith('.png'):
            lightning_logs = item / "lightning_logs"
            if lightning_logs.exists() and lightning_logs.is_dir():
                runs.append(item)
    return sorted(runs)


def generate_label_from_dir_name(dir_name: str) -> str:
    """Generate a clean label from directory name."""
    # Convert snake_case or kebab-case to Title Case
    label = dir_name.replace("_", " ").replace("-", " ")
    label = " ".join(word.capitalize() for word in label.split())
    
    # Add descriptive labels for common patterns
    if "baseline" in dir_name.lower():
        return "Baseline Test"
    elif "both" in dir_name.lower():
        return "Both Augmentations"
    elif "gaussian" in dir_name.lower() or "noise" in dir_name.lower():
        return "Gaussian Noise"
    elif "time" in dir_name.lower() and "warp" in dir_name.lower():
        return "Time Warping"
    elif "channel" in dir_name.lower():
        return "Channel Ablation"
    else:
        return label


def main():
    """Generate comparison plot for all runs in augs directory and time_warp."""
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    augs_dir = project_root / "logs" / "augs"
    time_warp_dir = project_root / "logs" / "time_warp"
    
    if not augs_dir.exists():
        print(f"Error: augs directory not found: {augs_dir}")
        sys.exit(1)
    
    # Find all runs in augs directory
    runs = find_runs_with_lightning_logs(augs_dir)
    
    # Also check if time_warp directory exists and has lightning_logs
    if time_warp_dir.exists():
        time_warp_lightning = time_warp_dir / "lightning_logs"
        if time_warp_lightning.exists() and time_warp_lightning.is_dir():
            runs.append(time_warp_dir)
    
    if not runs:
        print(f"No runs with lightning_logs found")
        sys.exit(0)
    
    print(f"Found {len(runs)} runs")
    print("=" * 80)
    
    # Prepare run configurations with labels
    run_configs = []
    for run_dir in runs:
        label = generate_label_from_dir_name(run_dir.name)
        run_configs.append((run_dir, label))
        print(f"  - {run_dir.name} -> {label}")
    
    # Generate comparison plot
    output_path = augs_dir / "augs_comparison_cer_history.png"
    print(f"\nGenerating comparison plot...")
    print(f"  Output: {output_path}")
    
    plot_comparison(
        run_configs=run_configs,
        metric_name="CER",
        save_path=output_path,
        ylim_max=100.0,
        plot_title="Augmentation Configurations Comparison - Validation CER",
        phase="val"  # Only show validation to keep it clean
    )
    
    print(f"\n✓ Comparison plot generated successfully!")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
