# code/0217_opt_plot.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR = PROJECT_ROOT / "data" / "output" / "opt_npz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "opt_plot"


def _to_scalar(x):
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        if x.size == 1:
            return x.reshape(-1)[0].item() if hasattr(x.reshape(-1)[0], "item") else x.reshape(-1)[0]
    return x


def _fmt_float(x, ndigits: int = 6) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _fmt_bool(x) -> str:
    try:
        return str(bool(x))
    except Exception:
        return str(x)


def build_info_lines(data: np.lib.npyio.NpzFile) -> list[str]:
    subject = _to_scalar(data["subject"]) if "subject" in data else "N/A"
    hemisphere = _to_scalar(data["hemisphere"]) if "hemisphere" in data else "N/A"
    target_name = _to_scalar(data["target_name"]) if "target_name" in data else "N/A"
    loss_name = _to_scalar(data["loss_name"]) if "loss_name" in data else "N/A"
    threshold = _to_scalar(data["threshold"]) if "threshold" in data else np.nan

    best_x = data["best_x"] if "best_x" in data else np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    alpha = best_x[0] if best_x.size > 0 else np.nan
    beta = best_x[1] if best_x.size > 1 else np.nan
    offset = best_x[2] if best_x.size > 2 else np.nan
    shank = best_x[3] if best_x.size > 3 else np.nan

    best_fun = _to_scalar(data["best_fun"]) if "best_fun" in data else np.nan
    dice = _to_scalar(data["dice"]) if "dice" in data else np.nan
    hell_d = _to_scalar(data["hell_d"]) if "hell_d" in data else np.nan
    grid_yield = _to_scalar(data["grid_yield"]) if "grid_yield" in data else np.nan
    grid_valid = _to_scalar(data["grid_valid"]) if "grid_valid" in data else False
    n_calls = _to_scalar(data["n_calls"]) if "n_calls" in data else -1

    contacts = data["contacts_xyz_moved"] if "contacts_xyz_moved" in data else np.empty((3, 0))
    n_contacts = contacts.shape[1] if contacts.ndim == 2 else 0

    opt_sec = _to_scalar(data["optimization_elapsed_seconds"]) if "optimization_elapsed_seconds" in data else np.nan
    sample_sec = _to_scalar(data["sample_elapsed_seconds"]) if "sample_elapsed_seconds" in data else np.nan

    lines = [
        f"Subject: {subject}",
        f"Hemisphere: {hemisphere}",
        f"Target map: {target_name}",
        f"Loss name: {loss_name}",
        f"Threshold: {_fmt_float(threshold, 4)}",
        "",
        "Best parameters",
        f"  alpha: {_fmt_float(alpha, 4)}",
        f"  beta: {_fmt_float(beta, 4)}",
        f"  offset_from_base: {_fmt_float(offset, 4)}",
        f"  shank_length: {_fmt_float(shank, 4)}",
        "",
        "Optimization result",
        f"  best_fun(loss): {_fmt_float(best_fun, 6)}",
        f"  dice: {_fmt_float(dice, 6)}",
        f"  hell_d: {_fmt_float(hell_d, 6)}",
        f"  grid_yield: {_fmt_float(grid_yield, 6)}",
        f"  grid_valid: {_fmt_bool(grid_valid)}",
        f"  n_calls: {n_calls}",
        f"  n_contacts: {n_contacts}",
        "",
        "Timing",
        f"  optimization_elapsed_sec: {_fmt_float(opt_sec, 2)}",
        f"  sample_elapsed_sec: {_fmt_float(sample_sec, 2)}",
    ]
    return lines


def plot_from_npz(npz_path: Path, output_dir: Path) -> Path:
    with np.load(npz_path, allow_pickle=True) as data:
        target_map = data["target_density"]
        predicted_map = data["phosphene_map"]
        info_lines = build_info_lines(data)

    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.05)

    ax_target = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[0, 2])

    ax_target.imshow(target_map, cmap="seismic")
    ax_target.set_title("Target map", fontsize=13)
    ax_target.axis("off")

    ax_pred.imshow(predicted_map, cmap="seismic")
    ax_pred.set_title("Predicted map", fontsize=13)
    ax_pred.axis("off")

    ax_info.axis("off")
    y = 0.98
    for line in info_lines:
        ax_info.text(0.0, y, line, va="top", ha="left", fontsize=10, family="monospace")
        y -= 0.046 if line != "" else 0.03

    fig.suptitle(npz_path.stem, fontsize=11, y=0.99)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{npz_path.stem}_report.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    npz_files = sorted(NPZ_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[skip] no npz files found in: {NPZ_DIR}")
        return

    print(f"found npz files: {len(npz_files)}")
    for npz_path in npz_files:
        out_path = plot_from_npz(npz_path, OUTPUT_DIR)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
