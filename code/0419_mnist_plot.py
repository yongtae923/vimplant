"""Visualize optimization results from 0415_mnist_opt.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# Ensure basecode modules are available
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASECODE_DIR = PROJECT_ROOT / "basecode"
if BASECODE_DIR.exists():
    sys.path.insert(0, str(BASECODE_DIR))

# Configuration
INPUT_DIR = PROJECT_ROOT / "data" / "output" / "opt_npz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "0419_mnist_opt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP = cm.get_cmap("viridis")


def plot_optimization_result(npz_file: Path, output_dir: Path) -> None:
    """Plot phosphene map and metrics from NPZ result file."""
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"[skip] Failed to load {npz_file.name}: {e}")
        return

    # Extract metadata
    subject = str(data["subject"])
    hemisphere = str(data["hemisphere"])
    target_name = str(data["target_name"])
    loss_name = str(data["loss_name"])
    
    # Extract metrics
    dice = float(data["dice"])
    grid_yield = float(data["grid_yield"])
    hell_d = float(data["hell_d"])
    best_fun = float(data["best_fun"])
    n_calls = int(data["n_calls"])
    optimization_elapsed = float(data["optimization_elapsed_seconds"])
    
    # Extract maps
    phosphene_map = np.asarray(data["phosphene_map"])
    target_density = np.asarray(data["target_density"])
    
    # Extract coordinate data
    best_x = np.asarray(data["best_x"])
    x_iters = np.asarray(data["x_iters"])
    func_vals = np.asarray(data["func_vals"])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Target phosphene map
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(target_density, cmap=CMAP, origin="lower")
    ax1.set_title(f"Target Phosphene Map\n{target_name}")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=ax1)
    
    # 2. Optimized phosphene map
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(phosphene_map, cmap=CMAP, origin="lower")
    ax2.set_title(f"Optimized Phosphene Map\n{hemisphere} Hemisphere")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=ax2)
    
    # 3. Difference map
    ax3 = plt.subplot(2, 3, 3)
    diff = np.abs(target_density - phosphene_map)
    im3 = ax3.imshow(diff, cmap="hot", origin="lower")
    ax3.set_title("Absolute Difference")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y (pixels)")
    plt.colorbar(im3, ax=ax3)
    
    # 4. Optimization history
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(func_vals, "b-", linewidth=1.5, alpha=0.7)
    ax4.axhline(best_fun, color="r", linestyle="--", label=f"Best: {best_fun:.4f}")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Loss")
    ax4.set_title("Optimization History")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Metrics summary (text)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("off")
    metrics_text = f"""
    Subject: {subject}
    Hemisphere: {hemisphere}
    Target: {target_name}
    Loss: {loss_name}
    
    ━━ OPTIMIZATION ━━
    Iterations: {n_calls}
    Time: {optimization_elapsed:.2f} sec
    
    ━━ BEST PARAMETERS ━━
    Alpha: {best_x[0]:.1f}°
    Beta: {best_x[1]:.1f}°
    Offset: {best_x[2]:.1f} mm
    Shank: {best_x[3]:.1f} mm
    
    ━━ METRICS ━━
    Dice: {dice:.4f}
    Yield: {grid_yield:.4f}
    Hellinger: {hell_d:.4f}
    Loss: {best_fun:.4f}
    """
    ax5.text(
        0.1, 0.5, metrics_text,
        fontsize=10, family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    # 6. 2D parameter space (alpha vs beta)
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(
        x_iters[:, 0], x_iters[:, 1],
        c=func_vals, cmap="RdYlGn_r", s=30, alpha=0.6, edgecolors="k", linewidth=0.5
    )
    ax6.scatter(best_x[0], best_x[1], color="red", s=200, marker="*", 
                edgecolors="black", linewidth=1.5, label="Best", zorder=5)
    ax6.set_xlabel("Alpha (degrees)")
    ax6.set_ylabel("Beta (degrees)")
    ax6.set_title("Parameter Space: Alpha vs Beta")
    ax6.legend()
    plt.colorbar(scatter, ax=ax6, label="Loss")
    
    # Main title
    fig.suptitle(
        f"Optimization Results: {subject} {hemisphere}",
        fontsize=14, fontweight="bold"
    )
    
    # Save figure
    output_filename = f"{subject}_{hemisphere}_{target_name}_{loss_name}.png"
    output_path = output_dir / output_filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")
    return output_path


def plot_comparison_all(output_dir: Path, input_dir: Path) -> None:
    """Create comparison plot for all results if multiple exist."""
    npz_files = list(input_dir.glob("*.npz"))
    
    if len(npz_files) < 2:
        return
    
    fig, axes = plt.subplots(
        len(npz_files), 3,
        figsize=(12, 4 * len(npz_files))
    )
    
    if len(npz_files) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, npz_file in enumerate(sorted(npz_files)):
        try:
            data = np.load(npz_file, allow_pickle=True)
            target_density = np.asarray(data["target_density"])
            phosphene_map = np.asarray(data["phosphene_map"])
            dice = float(data["dice"])
            hell_d = float(data["hell_d"])
            subject = str(data["subject"])
            hemisphere = str(data["hemisphere"])
        except Exception as e:
            print(f"[skip] Failed to load {npz_file.name}: {e}")
            continue
        
        # Target
        im = axes[idx, 0].imshow(target_density, cmap=CMAP, origin="lower")
        axes[idx, 0].set_title(f"{subject} {hemisphere}\nTarget")
        axes[idx, 0].set_ylabel("Y (px)")
        
        # Optimized
        im = axes[idx, 1].imshow(phosphene_map, cmap=CMAP, origin="lower")
        axes[idx, 1].set_title(f"Optimized\nDice: {dice:.3f}")
        
        # Difference
        diff = np.abs(target_density - phosphene_map)
        im = axes[idx, 2].imshow(diff, cmap="hot", origin="lower")
        axes[idx, 2].set_title(f"Difference\nHD: {hell_d:.3f}")
        
        axes[idx, 0].set_xticks([])
        axes[idx, 1].set_xticks([])
        axes[idx, 2].set_xticks([])
    
    fig.suptitle("Optimization Results Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    comparison_path = output_dir / "comparison_all.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved comparison: {comparison_path.name}")


def main() -> None:
    """Main visualization routine."""
    if not INPUT_DIR.exists():
        print(f"[error] Input directory not found: {INPUT_DIR}")
        return
    
    npz_files = list(INPUT_DIR.glob("*.npz"))
    
    if not npz_files:
        print(f"[warning] No NPZ files found in {INPUT_DIR}")
        print("Run 0415_mnist_opt.py first to generate results.")
        return
    
    print(f"Found {len(npz_files)} result file(s) in {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Process each NPZ file
    for npz_file in tqdm(sorted(npz_files), desc="Processing results"):
        plot_optimization_result(npz_file, OUTPUT_DIR)
    
    # Create comparison plot if multiple files
    if len(npz_files) > 1:
        print()
        plot_comparison_all(OUTPUT_DIR, INPUT_DIR)
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
