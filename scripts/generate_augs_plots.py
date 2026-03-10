#!/usr/bin/env python3
"""
Generate PNG plots for all runs in the augs directory using view_cer_history.py

Usage:
    python scripts/generate_augs_plots.py
"""

import subprocess
import sys
from pathlib import Path


def find_runs_with_lightning_logs(base_dir: Path):
    """Find all directories that contain lightning_logs."""
    runs = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.endswith('.png'):
            lightning_logs = item / "lightning_logs"
            if lightning_logs.exists() and lightning_logs.is_dir():
                runs.append(item)
    return sorted(runs)


def generate_title_from_dir_name(dir_name: str) -> str:
    """Generate a descriptive title from directory name."""
    # Convert snake_case or kebab-case to Title Case
    title = dir_name.replace("_", " ").replace("-", " ")
    title = " ".join(word.capitalize() for word in title.split())
    
    # Add descriptive suffixes for common patterns
    if "baseline" in dir_name.lower():
        return f"Baseline Test - CER History"
    elif "both" in dir_name.lower():
        return f"Both Augmentations - CER History"
    elif "gaussian" in dir_name.lower() or "noise" in dir_name.lower():
        return f"Gaussian Noise Augmentation - CER History"
    elif "time" in dir_name.lower() and "warp" in dir_name.lower():
        return f"Time Warping Augmentation - CER History"
    elif "channel" in dir_name.lower():
        return f"Channel Ablation - CER History"
    else:
        return f"{title} - CER History"


def generate_plot_for_run(run_dir: Path, script_path: Path, use_conda: bool = True, ylim_max: float = None, plot_title: str = None):
    """Generate PNG plot for a single run."""
    # Create output filename based on run directory name
    output_filename = f"{run_dir.name}_cer_history.png"
    output_path = run_dir / output_filename
    
    print(f"Processing: {run_dir.name}")
    print(f"  Input: {run_dir}")
    print(f"  Output: {output_path}")
    
    # Run view_cer_history.py script
    cmd_base = [
        sys.executable,
        str(script_path),
        str(run_dir),
        "--plot",
        "--save-plot",
        str(output_path),
    ]
    
    # Add ylim-max if specified
    if ylim_max is not None:
        cmd_base.extend(["--ylim-max", str(ylim_max)])
    
    # Add title if specified
    if plot_title:
        cmd_base.extend(["--title", plot_title])
    
    if use_conda:
        # Use conda run to ensure we're in the right environment
        cmd = ["conda", "run", "-n", "emg2qwerty"] + cmd_base
    else:
        cmd = cmd_base
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ Successfully generated plot\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to generate plot")
        print(f"    Error: {e.stderr}\n")
        return False


def main():
    """Generate PNG plots for all runs in augs directory and time_warp."""
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    augs_dir = project_root / "logs" / "augs"
    time_warp_dir = project_root / "logs" / "time_warp"
    view_cer_script = script_dir / "view_cer_history.py"
    
    if not augs_dir.exists():
        print(f"Error: augs directory not found: {augs_dir}")
        sys.exit(1)
    
    if not view_cer_script.exists():
        print(f"Error: view_cer_history.py not found: {view_cer_script}")
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
    
    # Generate plots for each run
    results = []
    for run_dir in runs:
        # Generate descriptive title
        plot_title = generate_title_from_dir_name(run_dir.name)
        # Set y-axis max to 100 to zoom in (don't show starting high CER)
        success = generate_plot_for_run(
            run_dir, 
            view_cer_script, 
            use_conda=True,
            ylim_max=100.0,
            plot_title=plot_title
        )
        results.append((run_dir.name, success))
    
    # Print summary
    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    for run_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{run_name}: {status}")
    
    # Check if all succeeded
    all_success = all(success for _, success in results)
    if not all_success:
        print("\n⚠ Some plots failed to generate. Check the errors above.")
        sys.exit(1)
    else:
        print("\n✓ All plots generated successfully!")
        
        # Also generate comparison plot
        print("\n" + "=" * 80)
        print("Generating comparison plot...")
        print("=" * 80)
        try:
            comparison_script = script_dir / "generate_augs_comparison.py"
            if comparison_script.exists():
                import subprocess
                cmd = ["conda", "run", "-n", "emg2qwerty", sys.executable, str(comparison_script)]
                subprocess.run(cmd, check=True)
            else:
                print("Comparison script not found, skipping...")
        except Exception as e:
            print(f"Warning: Could not generate comparison plot: {e}")
        
        sys.exit(0)


if __name__ == "__main__":
    main()
