# code/0217_opt_select.py
"""
Greedy selection strategy:
- Start with no active contacts.
- At each step, test every remaining contact by adding exactly one contact.
- Select the contact that gives the minimum trial loss (argmin).
- Stop when improvement becomes too small after enough channels are selected.
"""
from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure subprocess workers also suppress this warning.
_worker_warning_rule = "ignore::UserWarning:numpy._core.getlimits"
_py_warnings = os.environ.get("PYTHONWARNINGS", "")
if _worker_warning_rule not in _py_warnings.split(","):
    os.environ["PYTHONWARNINGS"] = f"{_py_warnings},{_worker_warning_rule}".strip(",")

warnings.filterwarnings(
    "ignore",
    message=r"Signature .* for <class 'numpy\.longdouble'> does not match any known type.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASECODE_DIR = PROJECT_ROOT / "basecode"
if BASECODE_DIR.exists():
    sys.path.insert(0, str(BASECODE_DIR))
else:
    raise FileNotFoundError(f"basecode directory not found: {BASECODE_DIR}")

from electphos import prf_to_phos
from lossfunc import DC, get_yield, hellinger_distance


INPUT_NPZ_DIR = PROJECT_ROOT / "data" / "output" / "opt_npz"
OUTPUT_NPZ_DIR = PROJECT_ROOT / "data" / "output" / "opt_select_npz"
OUTPUT_PLOT_DIR = PROJECT_ROOT / "data" / "output" / "opt_select_plot"

TOTAL_CORES = os.cpu_count() or 1
MAX_WORKERS = max(1, int(TOTAL_CORES * 0.9))

WINDOWSIZE = 1000
VIEW_ANGLE = 90
DC_PERCENTILE = 50

WEIGHT_A = 1.0
WEIGHT_B = 0.1
WEIGHT_C = 1.0
PENALTY = 0.25

MIN_IMPROVEMENT = 1e-6
MIN_ACTIVE_BEFORE_STOP = 50
OVERWRITE = False


def normalize_probability_map(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    maxv = float(out.max()) if out.size > 0 else 0.0
    if maxv > 0:
        out /= maxv
    sumv = float(out.sum())
    if sumv > 0:
        out /= sumv
    return out


def render_map_from_subset(phosphenes_v1: np.ndarray, active_idx: list[int]) -> np.ndarray:
    canvas = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype=np.float32)
    if not active_idx:
        return canvas
    subset = phosphenes_v1[np.array(active_idx, dtype=int)]
    canvas = prf_to_phos(canvas, subset, view_angle=VIEW_ANGLE, phSizeScale=1)
    return normalize_probability_map(canvas)


def subset_contacts(contacts_xyz_moved: np.ndarray, indices: list[int]) -> np.ndarray:
    arr = np.asarray(contacts_xyz_moved)
    if len(indices) == 0:
        if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
            return arr[:, []]
        return arr[[]]

    idx = np.array(indices, dtype=int)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        return arr[:, idx]
    return arr[idx]


def compute_loss(
    target_density: np.ndarray,
    phos_map: np.ndarray,
    contacts_subset: np.ndarray,
    good_coords: np.ndarray,
    grid_valid: bool,
    a: float = WEIGHT_A,
    b: float = WEIGHT_B,
    c: float = WEIGHT_C,
) -> tuple[float, dict[str, float]]:
    bin_thresh = np.percentile(target_density, DC_PERCENTILE)
    dice, _, _ = DC(target_density, phos_map, bin_thresh)
    par1 = 1.0 - a * dice

    try:
        grid_yield = float(get_yield(contacts_subset, good_coords))
    except Exception:
        grid_yield = 0.0
    par2 = 1.0 - b * grid_yield

    hell = hellinger_distance(phos_map.flatten(), target_density.flatten())
    if np.isnan(hell) or np.isinf(hell):
        hell = 1.0
    par3 = c * hell

    cost = par1 + par2 + par3
    if not bool(grid_valid):
        cost = par1 + PENALTY + par2 + PENALTY + par3 + PENALTY
    if np.isnan(cost) or np.isinf(cost):
        cost = 3.0

    return float(cost), {"dice": float(dice), "hell": float(hell), "yield": float(grid_yield)}


def greedy_select(
    phosphenes_v1: np.ndarray,
    contacts_xyz_moved: np.ndarray,
    good_coords: np.ndarray,
    grid_valid: bool,
    target_density: np.ndarray,
    min_improvement: float = MIN_IMPROVEMENT,
    min_active_before_stop: int = MIN_ACTIVE_BEFORE_STOP,
) -> tuple[list[int], np.ndarray, list[float], list[dict[str, float]]]:
    """
    Greedy strategy:
    - Start with no active contacts.
    - At each step, test every remaining contact by adding exactly one contact.
    - Select the contact that gives the minimum trial loss (argmin).
    - Stop when improvement becomes too small after enough channels are selected.
    """
    n = int(phosphenes_v1.shape[0])
    remaining = list(range(n))
    active: list[int] = []

    current_map = render_map_from_subset(phosphenes_v1, active)
    contacts_subset = subset_contacts(contacts_xyz_moved, active)
    best_loss, _ = compute_loss(target_density, current_map, contacts_subset, good_coords, grid_valid)

    history = [best_loss]
    progress: list[dict[str, float]] = []
    step = 0

    while remaining:
        best_i: int | None = None
        best_trial_loss: float | None = None
        best_comps: dict[str, float] | None = None

        for i in remaining:
            trial_active = active + [i]
            trial_map = render_map_from_subset(phosphenes_v1, trial_active)
            trial_contacts = subset_contacts(contacts_xyz_moved, trial_active)
            trial_loss, comps = compute_loss(target_density, trial_map, trial_contacts, good_coords, grid_valid)

            if (best_trial_loss is None) or (trial_loss < best_trial_loss):
                best_trial_loss = trial_loss
                best_i = i
                best_comps = comps

        if best_i is None or best_trial_loss is None or best_comps is None:
            break

        delta_best = best_loss - best_trial_loss
        if len(active) >= min_active_before_stop and delta_best <= min_improvement:
            break

        active.append(best_i)
        remaining.remove(best_i)

        current_map = render_map_from_subset(phosphenes_v1, active)
        contacts_subset = subset_contacts(contacts_xyz_moved, active)
        best_loss, comps = compute_loss(target_density, current_map, contacts_subset, good_coords, grid_valid)

        history.append(best_loss)
        progress.append(
            {
                "step": float(step),
                "picked_index": float(best_i),
                "active_count": float(len(active)),
                "loss": float(best_loss),
                "dice": float(comps["dice"]),
                "hell": float(comps["hell"]),
                "yield": float(comps["yield"]),
            }
        )
        step += 1

    return active, current_map, history, progress


def _fmt(x: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _save_select_report(
    plot_dir: Path,
    original_map: np.ndarray,
    selected_map: np.ndarray,
    data: dict[str, np.ndarray],
    orig_total: int,
    selected_count: int,
    full_loss: float,
    selected_loss: float,
    full_dice: float,
    selected_dice: float,
    full_hell: float,
    selected_hell: float,
    opt_elapsed: float,
    select_elapsed: float,
    stem: str,
) -> None:
    subject = _scalar_or_default(data, "subject", "N/A")
    hemisphere = _scalar_or_default(data, "hemisphere", "N/A")
    target_name = _scalar_or_default(data, "target_name", "N/A")

    lines = [
        f"Subject: {subject}",
        f"Hemisphere: {hemisphere}",
        f"Target map: {target_name}",
        "",
        "Channel count",
        f"  original: {orig_total}",
        f"  selected: {selected_count}",
        f"  (reduced to {selected_count}/{orig_total})",
        "",
        "Loss (opt full -> greedy selected)",
        f"  original: {_fmt(full_loss, 6)}",
        f"  selected: {_fmt(selected_loss, 6)}",
        "",
        "Dice",
        f"  original: {_fmt(full_dice, 6)}",
        f"  selected: {_fmt(selected_dice, 6)}",
        "",
        "Hellinger distance",
        f"  original: {_fmt(full_hell, 6)}",
        f"  selected: {_fmt(selected_hell, 6)}",
        "",
        "Timing (sec)",
        f"  opt: {_fmt(opt_elapsed, 2)}",
        f"  select (additional): {_fmt(select_elapsed, 2)}",
    ]

    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.05)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_sel = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[0, 2])

    ax_orig.imshow(original_map, cmap="seismic")
    ax_orig.set_title("Original map", fontsize=13)
    ax_orig.axis("off")

    ax_sel.imshow(selected_map, cmap="seismic")
    ax_sel.set_title("Selected map", fontsize=13)
    ax_sel.axis("off")

    ax_info.axis("off")
    y = 0.98
    for line in lines:
        ax_info.text(0.0, y, line, va="top", ha="left", fontsize=10, family="monospace")
        y -= 0.046 if line != "" else 0.03

    fig.suptitle(stem, fontsize=11, y=0.99)
    plot_dir.mkdir(parents=True, exist_ok=True)
    png_path = plot_dir / f"{stem}_report.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _scalar_or_default(data: dict[str, np.ndarray], key: str, default: Any) -> Any:
    if key not in data:
        return default
    val = data[key]
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    if isinstance(val, np.ndarray) and val.size == 1:
        return val.reshape(-1)[0].item() if hasattr(val.reshape(-1)[0], "item") else val.reshape(-1)[0]
    return val


def process_one_npz(npz_path: Path) -> tuple[str, bool, str]:
    sample_t0 = perf_counter()
    out_path = OUTPUT_NPZ_DIR / npz_path.name
    if out_path.exists() and not OVERWRITE:
        return (npz_path.name, True, "skip_exists")

    with np.load(npz_path, allow_pickle=True) as src:
        data = {k: src[k] for k in src.files}

    required = ["phosphenes_V1", "contacts_xyz_moved", "good_coords", "grid_valid", "target_density"]
    missing = [k for k in required if k not in data]
    if missing:
        return (npz_path.name, False, f"missing_keys:{','.join(missing)}")

    phosphenes_v1 = np.asarray(data["phosphenes_V1"])
    contacts_xyz_moved = np.asarray(data["contacts_xyz_moved"])
    good_coords = np.asarray(data["good_coords"])
    grid_valid = bool(_scalar_or_default(data, "grid_valid", True))
    target_density = normalize_probability_map(np.asarray(data["target_density"], dtype=np.float64))

    greedy_t0 = perf_counter()
    active_idx, selected_map, loss_history, progress = greedy_select(
        phosphenes_v1=phosphenes_v1,
        contacts_xyz_moved=contacts_xyz_moved,
        good_coords=good_coords,
        grid_valid=grid_valid,
        target_density=target_density,
    )
    greedy_elapsed = perf_counter() - greedy_t0

    full_idx = list(range(int(phosphenes_v1.shape[0])))
    full_map = render_map_from_subset(phosphenes_v1, full_idx)
    selected_contacts = subset_contacts(contacts_xyz_moved, active_idx)

    selected_loss, selected_comps = compute_loss(
        target_density, selected_map, selected_contacts, good_coords, grid_valid
    )
    full_loss, full_comps = compute_loss(
        target_density, full_map, contacts_xyz_moved, good_coords, grid_valid
    )

    # Keep all original keys, then append greedy outputs.
    out = dict(data)
    out.update(
        {
            "greedy_selected_indices": np.asarray(active_idx, dtype=np.int32),
            "greedy_selected_map": np.asarray(selected_map, dtype=np.float32),
            "greedy_full_map": np.asarray(full_map, dtype=np.float32),
            "greedy_loss_history": np.asarray(loss_history, dtype=np.float64),
            "greedy_progress_step": np.asarray([p["step"] for p in progress], dtype=np.float64),
            "greedy_progress_picked_index": np.asarray([p["picked_index"] for p in progress], dtype=np.float64),
            "greedy_progress_active_count": np.asarray([p["active_count"] for p in progress], dtype=np.float64),
            "greedy_progress_loss": np.asarray([p["loss"] for p in progress], dtype=np.float64),
            "greedy_progress_dice": np.asarray([p["dice"] for p in progress], dtype=np.float64),
            "greedy_progress_hell": np.asarray([p["hell"] for p in progress], dtype=np.float64),
            "greedy_progress_yield": np.asarray([p["yield"] for p in progress], dtype=np.float64),
            "greedy_original_total_count": np.array(int(phosphenes_v1.shape[0]), dtype=np.int32),
            "greedy_selected_count": np.array(len(active_idx), dtype=np.int32),
            "greedy_selected_loss": np.array(selected_loss, dtype=np.float64),
            "greedy_selected_dice": np.array(selected_comps["dice"], dtype=np.float64),
            "greedy_selected_hell_d": np.array(selected_comps["hell"], dtype=np.float64),
            "greedy_selected_grid_yield": np.array(selected_comps["yield"], dtype=np.float64),
            "greedy_full_loss": np.array(full_loss, dtype=np.float64),
            "greedy_full_dice": np.array(full_comps["dice"], dtype=np.float64),
            "greedy_full_hell_d": np.array(full_comps["hell"], dtype=np.float64),
            "greedy_full_grid_yield": np.array(full_comps["yield"], dtype=np.float64),
            "greedy_elapsed_seconds": np.array(greedy_elapsed, dtype=np.float64),
            "greedy_sample_elapsed_seconds": np.array((perf_counter() - sample_t0), dtype=np.float64),
        }
    )

    OUTPUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    opt_elapsed = float(_scalar_or_default(data, "optimization_elapsed_seconds", 0.0))
    _save_select_report(
        plot_dir=OUTPUT_PLOT_DIR,
        original_map=np.asarray(data["phosphene_map"]),
        selected_map=selected_map,
        data=data,
        orig_total=int(phosphenes_v1.shape[0]),
        selected_count=len(active_idx),
        full_loss=full_loss,
        selected_loss=selected_loss,
        full_dice=full_comps["dice"],
        selected_dice=selected_comps["dice"],
        full_hell=full_comps["hell"],
        selected_hell=selected_comps["hell"],
        opt_elapsed=opt_elapsed,
        select_elapsed=greedy_elapsed,
        stem=out_path.stem,
    )
    return (npz_path.name, True, "saved")


def main() -> None:
    run_t0 = perf_counter()
    if not INPUT_NPZ_DIR.exists():
        print(f"[skip] input dir not found: {INPUT_NPZ_DIR}")
        return

    npz_files = sorted(INPUT_NPZ_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[skip] no npz files found in: {INPUT_NPZ_DIR}")
        return

    print(f"input npz dir: {INPUT_NPZ_DIR}")
    print(f"output npz dir: {OUTPUT_NPZ_DIR}")
    print(f"overwrite: {OVERWRITE}")
    print(f"input npz count: {len(npz_files)}")
    print(f"cpu cores: total={TOTAL_CORES}, workers={MAX_WORKERS} (90%)")
    print("greedy strategy: at each step choose one contact with minimum trial loss")

    ok = 0
    fail = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one_npz, p): p for p in npz_files}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                name, success, status = fut.result()
                if success:
                    ok += 1
                    print(f"[ok] {name} ({status})")
                else:
                    fail += 1
                    print(f"[fail] {name} ({status})")
            except Exception as e:
                fail += 1
                print(f"[fail] {p.name} ({e})")

    total_elapsed = perf_counter() - run_t0
    print(f"done: ok={ok}, fail={fail}, total_elapsed_seconds={total_elapsed:.2f}")


if __name__ == "__main__":
    main()
