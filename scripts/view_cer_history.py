#!/usr/bin/env python3
"""
Script to view CER (Character Error Rate) history per epoch from TensorBoard logs.

Usage:
    python scripts/view_cer_history.py <log_directory>
    
Example:
    python scripts/view_cer_history.py logs/2026-03-09/12-32-51
"""

import argparse
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard is not installed. Install it with: pip install tensorboard")
    sys.exit(1)


def extract_cer_history(log_dir: Path, metric_name: str = "CER"):
    """
    Extract CER history from TensorBoard events.
    
    Args:
        log_dir: Directory containing lightning_logs
        metric_name: Name of the metric to extract (default: "CER")
    
    Returns:
        Dictionary with keys: 'train', 'val', 'test'
        Each value is a list of tuples: (step, wall_time, value)
    """
    # Find the lightning_logs directory
    lightning_logs_dir = log_dir / "lightning_logs"
    if not lightning_logs_dir.exists():
        # Try direct path
        if log_dir.name == "lightning_logs":
            lightning_logs_dir = log_dir
        else:
            raise FileNotFoundError(
                f"Could not find lightning_logs directory in {log_dir}"
            )
    
    # Find version_0 directory
    version_dirs = list(lightning_logs_dir.glob("version_*"))
    if not version_dirs:
        raise FileNotFoundError(
            f"Could not find version_* directory in {lightning_logs_dir}"
        )
    
    # Use the first version directory (usually version_0)
    version_dir = sorted(version_dirs)[0]
    
    # Find events file
    events_files = list(version_dir.glob("events.out.tfevents.*"))
    if not events_files:
        raise FileNotFoundError(f"Could not find events file in {version_dir}")
    
    # Load events
    ea = EventAccumulator(str(version_dir))
    ea.Reload()
    
    # Check available tags
    if "scalars" not in ea.Tags():
        raise ValueError("No scalar metrics found in TensorBoard logs")
    
    # Extract metrics
    metrics = {}
    for phase in ["train", "val", "test"]:
        metric_key = f"{phase}/{metric_name}"
        if metric_key in ea.Tags()["scalars"]:
            scalar_events = ea.Scalars(metric_key)
            metrics[phase] = [
                (event.step, event.wall_time, event.value)
                for event in scalar_events
            ]
        else:
            metrics[phase] = []
    
    # Try to get epoch information if available
    epoch_info = {}
    if "epoch" in ea.Tags()["scalars"]:
        epoch_events = ea.Scalars("epoch")
        # Map step to epoch
        for event in epoch_events:
            epoch_info[int(event.step)] = int(event.value)
    
    return metrics, epoch_info


def print_cer_table(metrics: dict, epoch_info: dict = None, metric_name: str = "CER"):
    """Print CER history as a table."""
    print(f"\n{'='*80}")
    print(f"{metric_name} History per Epoch")
    print(f"{'='*80}\n")
    
    # Use validation steps as epochs (validation runs once per epoch)
    if not metrics["val"]:
        print("No validation metrics found!")
        return
    
    # Get validation steps (these correspond to epochs)
    val_steps = sorted([step for step, _, _ in metrics["val"]])
    
    # If we have epoch info, use it; otherwise use step indices as epoch numbers
    if epoch_info:
        # Map steps to epochs
        step_to_epoch = {}
        for step in val_steps:
            # Find closest epoch event
            closest_epoch_step = min(epoch_info.keys(), key=lambda x: abs(x - step))
            step_to_epoch[step] = epoch_info[closest_epoch_step]
        epochs = [step_to_epoch.get(step, i) for i, step in enumerate(val_steps)]
    else:
        # Use step indices as epochs (0-indexed)
        epochs = list(range(len(val_steps)))
    
    # Print header
    print(f"{'Epoch':<8} {'Step':<10} {'Train':<12} {'Val':<12} {'Test':<12}")
    print("-" * 80)
    
    # Print each epoch
    for epoch, step in zip(epochs, val_steps):
        train_val = next(
            (val for s, _, val in metrics["train"] if s == step),
            None
        )
        val_val = next(
            (val for s, _, val in metrics["val"] if s == step),
            None
        )
        test_val = next(
            (val for s, _, val in metrics["test"] if s == step),
            None
        )
        
        train_str = f"{train_val:.4f}" if train_val is not None else "N/A"
        val_str = f"{val_val:.4f}" if val_val is not None else "N/A"
        test_str = f"{test_val:.4f}" if test_val is not None else "N/A"
        
        print(f"{epoch:<8} {step:<10} {train_str:<12} {val_str:<12} {test_str:<12}")
    
    print(f"\n{'='*80}\n")
    
    # Print summary statistics
    if metrics["val"]:
        val_values = [val for _, _, val in metrics["val"]]
        best_idx = val_values.index(min(val_values))
        worst_idx = val_values.index(max(val_values))
        print(f"Validation {metric_name} Summary:")
        print(f"  Best: {min(val_values):.4f} (Epoch {epochs[best_idx]}, Step {val_steps[best_idx]})")
        print(f"  Worst: {max(val_values):.4f} (Epoch {epochs[worst_idx]}, Step {val_steps[worst_idx]})")
        print(f"  Final: {val_values[-1]:.4f} (Epoch {epochs[-1]}, Step {val_steps[-1]})")
        print(f"  Mean: {sum(val_values)/len(val_values):.4f}")
        print()


def plot_cer_history(metrics: dict, epoch_info: dict = None, metric_name: str = "CER", save_path: Path = None, ylim_max: float = None, plot_title: str = None):
    """Plot CER history as a line chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot generation.")
        print("Install it with: pip install matplotlib")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Use validation steps to determine x-axis (epochs)
    if metrics["val"]:
        val_steps = sorted([step for step, _, _ in metrics["val"]])
        if epoch_info:
            # Map steps to epochs
            x_vals = [epoch_info.get(min(epoch_info.keys(), key=lambda x: abs(x - step)), i) 
                     for i, step in enumerate(val_steps)]
            x_label = "Epoch"
        else:
            x_vals = list(range(len(val_steps)))
            x_label = "Epoch (inferred)"
    else:
        x_vals = None
        x_label = "Step"
    
    for phase, color, label in [
        ("train", "blue", "Train"),
        ("val", "green", "Validation"),
        ("test", "red", "Test"),
    ]:
        if metrics[phase]:
            if phase == "val" and x_vals is not None:
                # For validation, use epoch-based x-axis
                steps = sorted([step for step, _, _ in metrics[phase]])
                values = [val for step, _, val in sorted(metrics[phase], key=lambda x: x[0])]
                plt.plot(x_vals[:len(values)], values, color=color, marker="o", label=label, linewidth=2)
            else:
                # For train/test, use steps or indices
                steps = [step for step, _, _ in metrics[phase]]
                values = [val for _, _, val in metrics[phase]]
                if x_vals and phase == "train":
                    # Try to align train metrics with validation epochs
                    # This is approximate - train metrics are logged more frequently
                    plt.plot(x_vals[:len(values)] if len(values) <= len(x_vals) else list(range(len(values))), 
                            values, color=color, marker=".", label=label, linewidth=1, alpha=0.7)
                else:
                    plt.plot(list(range(len(values))), values, color=color, marker="o", label=label, linewidth=2)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(f"{metric_name} (%)", fontsize=12)
    
    # Use custom title if provided, otherwise default
    if plot_title:
        plt.title(plot_title, fontsize=14, fontweight="bold")
    else:
        plt.title(f"{metric_name} History", fontsize=14, fontweight="bold")
    
    # Set y-axis limit if specified
    if ylim_max is not None:
        plt.ylim(0, ylim_max)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_comparison(run_configs: list[tuple[Path, str]], metric_name: str = "CER", save_path: Path = None, ylim_max: float = None, plot_title: str = None, phase: str = "val"):
    """
    Plot comparison of multiple runs on a single plot.
    
    Args:
        run_configs: List of tuples (log_dir, label) for each run to compare
        metric_name: Name of the metric to extract (default: "CER")
        save_path: Path to save the plot
        ylim_max: Maximum value for y-axis
        plot_title: Custom title for the plot
        phase: Which phase to plot ("val", "test", or "train") - default "val"
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib/numpy not installed. Skipping plot generation.")
        print("Install it with: pip install matplotlib numpy")
        return
    
    # Define distinct colors and line styles for up to 10 configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    plt.figure(figsize=(12, 7))
    
    all_x_vals = []
    all_epoch_infos = []
    
    # Extract data from all runs
    for idx, (run_dir, label) in enumerate(run_configs):
        try:
            metrics, epoch_info = extract_cer_history(run_dir, metric_name)
            
            if not metrics.get(phase, []):
                print(f"Warning: No {phase} metrics found for {label}, skipping...")
                continue
            
            # Get validation steps and values
            phase_steps = sorted([step for step, _, _ in metrics[phase]])
            phase_values = [val for step, _, val in sorted(metrics[phase], key=lambda x: x[0])]
            
            # Map steps to epochs
            if epoch_info:
                x_vals = []
                for step in phase_steps:
                    closest_epoch_step = min(epoch_info.keys(), key=lambda x: abs(x - step))
                    x_vals.append(epoch_info[closest_epoch_step])
            else:
                x_vals = list(range(len(phase_steps)))
            
            all_x_vals.append(x_vals)
            all_epoch_infos.append(epoch_info)
            
            # Plot this run
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            marker = markers[idx % len(markers)]
            
            plt.plot(
                x_vals, 
                phase_values, 
                color=color, 
                linestyle=linestyle,
                marker=marker,
                label=label, 
                linewidth=2.5,
                markersize=6,
                markevery=max(1, len(x_vals) // 20)  # Show markers every ~5% of points
            )
            
        except Exception as e:
            print(f"Error processing {label}: {e}")
            continue
    
    if not all_x_vals:
        print("Error: No valid data found to plot")
        return
    
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel(f"{metric_name} (%)", fontsize=13)
    
    # Use custom title if provided
    if plot_title:
        plt.title(plot_title, fontsize=15, fontweight="bold")
    else:
        phase_label = phase.capitalize()
        plt.title(f"{phase_label} {metric_name} Comparison", fontsize=15, fontweight="bold")
    
    # Set y-axis limit if specified
    if ylim_max is not None:
        plt.ylim(0, ylim_max)
    
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="View CER history from TensorBoard logs"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to log directory (e.g., logs/2026-03-09/12-32-51)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="CER",
        help="Metric name to extract (default: CER). Can also use IER, DER, SER",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a plot of the CER history",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save plot to file instead of displaying",
    )
    parser.add_argument(
        "--ylim-max",
        type=float,
        default=None,
        help="Maximum value for y-axis (e.g., 100 to zoom in)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)
    
    try:
        metrics, epoch_info = extract_cer_history(log_dir, args.metric)
        print_cer_table(metrics, epoch_info, args.metric)
        
        if args.plot or args.save_plot:
            plot_cer_history(
                metrics,
                epoch_info,
                metric_name=args.metric,
                save_path=Path(args.save_plot) if args.save_plot else None,
                ylim_max=args.ylim_max,
                plot_title=args.title,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
