# D:\yongtae\vimplant\code\0422_greedy_opt.py

from __future__ import annotations

import json
import csv
import concurrent.futures as cf
import os
import platform
import random
import sys
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence, cast

# Ensure subprocess workers (loky/multiprocessing) also suppress this warning.
_worker_warning_rule = "ignore::UserWarning:numpy._core.getlimits"
_py_warnings = os.environ.get("PYTHONWARNINGS", "")
if _worker_warning_rule not in _py_warnings.split(","):
    os.environ["PYTHONWARNINGS"] = f"{_py_warnings},{_worker_warning_rule}".strip(",")

# Hide known non-critical longdouble warning on some WSL/numpy builds.
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

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# Ensure basecode modules are always imported first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASECODE_DIR = PROJECT_ROOT / "basecode"
if BASECODE_DIR.exists():
    sys.path.insert(0, str(BASECODE_DIR))
else:
    raise FileNotFoundError(f"basecode directory not found: {BASECODE_DIR}")

# Local modules from basecode (same API as notebook)
from ninimplant import get_xyz
from lossfunc import DC as _DC, get_yield, hellinger_distance
from electphos import (
    cortical_spread,
    create_grid,
    get_cortical_magnification,
    get_phosphenes,
    implant_grid,
    prf_to_phos,
)

DC: Any = _DC


if not np.geterr().get("divide"):
    warnings.simplefilter("ignore")
np.seterr(divide="ignore", invalid="ignore")


# Hardcoded run configuration
SUBJECTS = ["100610"]
DATA_ROOT = PROJECT_ROOT / "data" / "input" / "100610"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output_greedy_0422"
TOTAL_CORES = os.cpu_count() or 1
# Use all available CPU cores for parallel workers
N_JOBS = max(1, TOTAL_CORES)
OVERWRITE = False
RESEARCH_SEED = 42
BASELINE_REPEATS = 5

# Optimization setup from notebook
DIM_ALPHA = Integer(name="alpha", low=-90, high=90)
DIM_BETA_LH = Integer(name="beta", low=-15, high=110)
DIM_BETA_RH = Integer(name="beta", low=-110, high=15)
DIM_OFFSET = Integer(name="offset_from_base", low=0, high=40)
DIM_SHANK = Integer(name="shank_length", low=10, high=40)

NUM_CALLS = 150
X0 = (0, 0, 20, 25)
NUM_INITIAL_POINTS = 10
DC_PERCENTILE = 50
N_CONTACTPOINTS_SHANK = 10
SPACING_ALONG_XY = 1
MIN_IMPROVEMENT = 1e-6
MIN_ACTIVE_BEFORE_STOP = 50

LOSS_COMB = [(1, 0.1, 1)]
LOSS_NAMES = ["dice-yield-HD"]
THRESH = 0.05

TARGET_DIR = PROJECT_ROOT / "data" / "targets0421"
if not TARGET_DIR.exists():
    raise FileNotFoundError(f"target directory not found: {TARGET_DIR}")

CORT_MAG_MODEL = "wedge-dipole"
VIEW_ANGLE = 90
AMP = 100

FNAME_ANG = "inferred_angle.mgz"
FNAME_ECC = "inferred_eccen.mgz"
FNAME_SIGMA = "inferred_sigma.mgz"
FNAME_APARC = "aparc+aseg.mgz"
FNAME_LABEL = "inferred_varea.mgz"


def normalize_density(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64, copy=True)
    max_val = np.max(arr)
    if max_val > 0:
        arr /= max_val
    total = np.sum(arr)
    if total > 0:
        arr /= total
    return arr


def normalize_probability_map(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    maxv = float(out.max()) if out.size > 0 else 0.0
    if maxv > 0:
        out /= maxv
    sumv = float(out.sum())
    if sumv > 0:
        out /= sumv
    return out


def load_nifti_fdata(path: Path) -> np.ndarray:
    img = cast(Any, nib).load(str(path))
    return np.asarray(img.get_fdata())


_GREEDY_POOL_CONTEXT: dict[str, Any] = {}


def _init_greedy_pool_context(
    phosphenes_v1: np.ndarray,
    contacts_xyz_moved: np.ndarray,
    good_coords: np.ndarray,
    grid_valid: bool,
    target_density: np.ndarray,
    target_shape: tuple[int, int],
) -> None:
    global _GREEDY_POOL_CONTEXT
    _GREEDY_POOL_CONTEXT = {
        "phosphenes_v1": phosphenes_v1,
        "contacts_xyz_moved": contacts_xyz_moved,
        "good_coords": good_coords,
        "grid_valid": grid_valid,
        "target_density": target_density,
        "target_shape": target_shape,
    }


def _evaluate_greedy_candidate(candidate_index: int, active_indices: tuple[int, ...]) -> tuple[int, float]:
    ctx = _GREEDY_POOL_CONTEXT
    trial_active = [idx for idx in active_indices if idx != candidate_index]
    trial_map = render_map_from_subset(ctx["phosphenes_v1"], trial_active, ctx["target_shape"])
    trial_contacts = subset_contacts(ctx["contacts_xyz_moved"], trial_active)
    trial_loss, _ = compute_loss(
        ctx["target_density"],
        trial_map,
        trial_contacts,
        ctx["good_coords"],
        ctx["grid_valid"],
    )
    return candidate_index, trial_loss


def render_map_from_subset(phosphenes_v1: np.ndarray, active_idx: list[int], target_shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(target_shape, dtype=np.float32)
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
    a: float = 1.0,
    b: float = 0.1,
    c: float = 1.0,
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
        cost += 0.25 * 3
    if np.isnan(cost) or np.isinf(cost):
        cost = 3.0

    return float(cost), {"dice": float(dice), "hell": float(hell), "yield": float(grid_yield)}


def greedy_select(
    phosphenes_v1: np.ndarray,
    contacts_xyz_moved: np.ndarray,
    good_coords: np.ndarray,
    grid_valid: bool,
    target_density: np.ndarray,
    target_shape: Sequence[int],
    min_improvement: float = MIN_IMPROVEMENT,
    min_active_before_stop: int = MIN_ACTIVE_BEFORE_STOP,
    n_jobs: int = N_JOBS,
) -> tuple[list[int], np.ndarray, list[float], list[dict[str, float]]]:
    n = int(phosphenes_v1.shape[0])
    active: list[int] = list(range(n))
    min_active_to_keep = max(1, min(int(min_active_before_stop), n))
    total_steps = max(0, n - min_active_to_keep)
    target_shape_2d = (int(target_shape[0]), int(target_shape[1]))

    current_map = render_map_from_subset(phosphenes_v1, active, target_shape_2d)
    contacts_subset = subset_contacts(contacts_xyz_moved, active)
    best_loss, _ = compute_loss(target_density, current_map, contacts_subset, good_coords, grid_valid)

    history = [best_loss]
    progress: list[dict[str, float]] = []
    step = 0

    pbar = tqdm(total=total_steps, desc="greedy prune", unit="step", leave=False)
    try:
        while len(active) > min_active_to_keep:
            active_tuple = tuple(active)
            best_i: int | None = None
            best_trial_loss: float | None = None

            if n_jobs > 1 and len(active_tuple) > 1:
                with cf.ProcessPoolExecutor(
                    max_workers=min(int(n_jobs), len(active_tuple)),
                    initializer=_init_greedy_pool_context,
                    initargs=(phosphenes_v1, contacts_xyz_moved, good_coords, grid_valid, target_density, target_shape_2d),
                ) as executor:
                    futures = [executor.submit(_evaluate_greedy_candidate, i, active_tuple) for i in active_tuple]
                    for future in cf.as_completed(futures):
                        candidate_index, trial_loss = future.result()
                        if best_trial_loss is None or trial_loss < best_trial_loss:
                            best_trial_loss = trial_loss
                            best_i = candidate_index
            else:
                for i in active:
                    trial_active = [idx for idx in active if idx != i]
                    trial_map = render_map_from_subset(phosphenes_v1, trial_active, target_shape_2d)
                    trial_contacts = subset_contacts(contacts_xyz_moved, trial_active)
                    trial_loss, _ = compute_loss(target_density, trial_map, trial_contacts, good_coords, grid_valid)

                    if (best_trial_loss is None) or (trial_loss < best_trial_loss):
                        best_trial_loss = trial_loss
                        best_i = i

            if best_i is None or best_trial_loss is None:
                break

            best_trial_active = [idx for idx in active if idx != best_i]
            best_trial_map = render_map_from_subset(phosphenes_v1, best_trial_active, target_shape_2d)
            best_trial_contacts = subset_contacts(contacts_xyz_moved, best_trial_active)
            best_trial_loss, best_comps = compute_loss(target_density, best_trial_map, best_trial_contacts, good_coords, grid_valid)

            delta_best = best_loss - best_trial_loss
            if delta_best <= min_improvement:
                break

            active = best_trial_active
            current_map = best_trial_map
            best_loss = best_trial_loss

            history.append(best_loss)
            progress.append(
                {
                    "step": float(step),
                    "picked_index": float(best_i),
                    "active_count": float(len(active)),
                    "loss": float(best_loss),
                    "dice": float(best_comps["dice"]),
                    "hell": float(best_comps["hell"]),
                    "yield": float(best_comps["yield"]),
                }
            )
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{best_loss:.4f}", active=len(active))
    finally:
        pbar.close()

    return active, current_map, history, progress


def coords_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((3, 0), dtype=np.int32)
    aset = set(map(tuple, np.round(a).T.astype(np.int32)))
    bset = set(map(tuple, np.round(b).T.astype(np.int32)))
    inter = list(aset & bset)
    if not inter:
        return np.empty((3, 0), dtype=np.int32)
    return np.array(inter, dtype=np.int32).T


def custom_stopper(res, n: int = 5, delta: float = 0.2, thresh: float = 0.05):
    """Early stop when top-N costs are stable and low."""
    if len(res.func_vals) < n:
        return None
    func_vals = np.sort(res.func_vals)
    worst = func_vals[n - 1]
    best = func_vals[0]
    if worst == 0:
        return best < thresh
    return (abs((best - worst) / worst) < delta) and (best < thresh)


def _coords_to_nx3(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"coords must be 2D, got shape={arr.shape}")
    if arr.shape[1] == 3:
        return arr.astype(np.float64, copy=False)
    if arr.shape[0] == 3:
        return arr.T.astype(np.float64, copy=False)
    raise ValueError(f"coords must be Nx3 or 3xN, got shape={arr.shape}")


def compute_electrode_activations(contacts_xyz_moved: np.ndarray, good_h: np.ndarray) -> np.ndarray:
    contacts = _coords_to_nx3(contacts_xyz_moved)
    if contacts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if np.asarray(good_h).size == 0:
        return np.zeros((contacts.shape[0],), dtype=np.float32)

    good_set = set(map(tuple, np.round(np.asarray(good_h).T).astype(np.int32)))
    activations = np.zeros((contacts.shape[0],), dtype=np.float32)
    for idx, c in enumerate(np.round(contacts).astype(np.int32)):
        activations[idx] = 1.0 if tuple(c.tolist()) in good_set else 0.0
    return activations


def save_density_plot(arr: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(arr, cmap="viridis", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_comparison_plot(target: np.ndarray, reconstruction: np.ndarray, out_path: Path) -> None:
    diff = np.abs(target - reconstruction)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(target, cmap="viridis", origin="lower")
    axes[0].set_title("Target")
    axes[0].axis("off")
    axes[1].imshow(reconstruction, cmap="viridis", origin="lower")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="magma", origin="lower")
    axes[2].set_title("Absolute Difference")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_bar_plot(values_by_label: dict[str, float], title: str, ylabel: str, out_path: Path) -> None:
    if not values_by_label:
        return
    labels = list(values_by_label.keys())
    values = [values_by_label[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4), dpi=150)
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_scatter_plot(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    if x.size == 0 or y.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.scatter(x, y, s=25, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_hist_plot(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 20) -> None:
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def collect_runtime_environment() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "nibabel_version": getattr(nib, "__version__", "unknown"),
        "matplotlib_version": getattr(matplotlib, "__version__", "unknown"),
        "cpu_count": int(os.cpu_count() or 1),
        "configured_jobs": int(N_JOBS),
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }


def compute_baseline_seconds(target_shape: Sequence[int], repeats: int = BASELINE_REPEATS) -> float:
    """Measure a small deterministic numeric workload for time normalization."""
    if len(target_shape) != 2:
        raise ValueError(f"target_shape must be 2D, got {tuple(target_shape)}")
    h, w = int(target_shape[0]), int(target_shape[1])
    arr1 = np.linspace(0.0, 1.0, h * w, dtype=np.float64).reshape(h, w)
    arr2 = np.flipud(arr1)

    elapsed: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = perf_counter()
        a = normalize_density(arr1)
        b = normalize_density(arr2)
        _ = hellinger_distance(a.ravel(), b.ravel())
        _ = float(np.sum(np.abs(a - b)))
        elapsed.append(perf_counter() - t0)
    return float(np.mean(elapsed))


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_aggregate_record_from_results(
    results_json_path: Path,
    run_dir: Path,
    fallback_target_name: str,
    fallback_subject: str,
    fallback_hemisphere: str,
) -> dict | None:
    try:
        with open(results_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] failed to parse existing result: {results_json_path} ({exc})")
        return None

    reconstruction_metrics = payload.get("reconstruction_metrics", {})
    composite_scores = payload.get("composite_scores", {})
    efficiency_metrics = payload.get("efficiency_metrics", {})
    greedy_metrics = payload.get("greedy_metrics", {})
    simulator_metrics = payload.get("simulator_metrics", {})
    timing_seconds = payload.get("timing_seconds", {})
    timing_normalized = payload.get("timing_normalized", {})
    optimization = payload.get("optimization", {})
    target = payload.get("target", {})

    return {
        "target_name": str(target.get("name", fallback_target_name)),
        "subject": str(payload.get("subject", fallback_subject)),
        "hemisphere": str(payload.get("hemisphere", fallback_hemisphere)),
        "run_dir": str(run_dir),
        "DC": _safe_float(reconstruction_metrics.get("DC")),
        "Y": _safe_float(reconstruction_metrics.get("Y")),
        "HD": _safe_float(reconstruction_metrics.get("HD")),
        "score": _safe_float(composite_scores.get("score")),
        "loss": _safe_float(composite_scores.get("loss")),
        "contact_count": _safe_int(efficiency_metrics.get("contact_count")),
        "active_count": _safe_int(efficiency_metrics.get("active_count")),
        "active_ratio": _safe_float(efficiency_metrics.get("active_ratio")),
        "selected_loss": _safe_float(greedy_metrics.get("selected_loss")),
        "selected_dice": _safe_float(greedy_metrics.get("selected_dice")),
        "selected_hell_d": _safe_float(greedy_metrics.get("selected_hell_d")),
        "selected_grid_yield": _safe_float(greedy_metrics.get("selected_grid_yield")),
        "greedy_elapsed_seconds": _safe_float(greedy_metrics.get("greedy_elapsed_seconds")),
        "simulator_forward_calls": _safe_int(simulator_metrics.get("simulator_forward_calls")),
        "model_forward_calls": _safe_int(simulator_metrics.get("model_forward_calls")),
        "simulator_calls_per_eval": _safe_float(simulator_metrics.get("simulator_calls_per_eval")),
        "model_calls_per_eval": _safe_float(simulator_metrics.get("model_calls_per_eval")),
        "wall_clock_time": _safe_float(timing_seconds.get("wall_clock_time")),
        "model_forward_time": _safe_float(timing_seconds.get("model_forward_time")),
        "simulator_forward_time": _safe_float(timing_seconds.get("simulator_forward_time")),
        "metric_computation_time": _safe_float(timing_seconds.get("metric_computation_time")),
        "greedy_elapsed_time": _safe_float(timing_seconds.get("greedy_elapsed_time")),
        "normalized_wall_clock_time": _safe_float(timing_normalized.get("normalized_wall_clock_time")),
        "normalized_greedy_elapsed_time": _safe_float(timing_normalized.get("normalized_greedy_elapsed_time")),
        "evals_per_second": _safe_float(timing_normalized.get("evals_per_second")),
        "model_forward_time_per_call": _safe_float(timing_normalized.get("model_forward_time_per_call")),
        "seed": _safe_int(optimization.get("seed", RESEARCH_SEED), default=int(RESEARCH_SEED)),
        "grid_valid": bool(payload.get("grid_valid", True)),
    }


def write_aggregate_outputs(records: list[dict], aggregate_dir: Path) -> None:
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    with open(aggregate_dir / "aggregate_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    np.save(aggregate_dir / "aggregate_results.npy", np.array(records, dtype=object), allow_pickle=True)

    if records:
        fieldnames = sorted(records[0].keys())
        with open(aggregate_dir / "aggregate_results.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    if not records:
        return

    summary = {
        "n_records": int(len(records)),
        "mean_DC": float(np.mean([float(r["DC"]) for r in records])),
        "std_DC": float(np.std([float(r["DC"]) for r in records])),
        "mean_Y": float(np.mean([float(r["Y"]) for r in records])),
        "std_Y": float(np.std([float(r["Y"]) for r in records])),
        "mean_HD": float(np.mean([float(r["HD"]) for r in records])),
        "std_HD": float(np.std([float(r["HD"]) for r in records])),
        "mean_score": float(np.mean([float(r["score"]) for r in records])),
        "std_score": float(np.std([float(r["score"]) for r in records])),
        "mean_loss": float(np.mean([float(r["loss"]) for r in records])),
        "std_loss": float(np.std([float(r["loss"]) for r in records])),
        "mean_contact_count": float(np.mean([float(r["contact_count"]) for r in records])),
        "std_contact_count": float(np.std([float(r["contact_count"]) for r in records])),
        "mean_active_count": float(np.mean([float(r["active_count"]) for r in records])),
        "std_active_count": float(np.std([float(r["active_count"]) for r in records])),
        "mean_selected_loss": float(np.mean([float(r["selected_loss"]) for r in records])),
        "std_selected_loss": float(np.std([float(r["selected_loss"]) for r in records])),
        "mean_greedy_elapsed": float(np.mean([float(r["greedy_elapsed_seconds"]) for r in records])),
        "std_greedy_elapsed": float(np.std([float(r["greedy_elapsed_seconds"]) for r in records])),
        "mean_norm_wall": float(np.mean([float(r["normalized_wall_clock_time"]) for r in records])),
        "std_norm_wall": float(np.std([float(r["normalized_wall_clock_time"]) for r in records])),
    }
    with open(aggregate_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    by_target_scores: dict[str, list[float]] = {}
    by_target_losses: dict[str, list[float]] = {}
    for r in records:
        tname = str(r["target_name"])
        by_target_scores.setdefault(tname, []).append(float(r["score"]))
        by_target_losses.setdefault(tname, []).append(float(r["loss"]))

    mean_score_by_target = {k: float(np.mean(v)) for k, v in by_target_scores.items()}
    mean_loss_by_target = {k: float(np.mean(v)) for k, v in by_target_losses.items()}
    save_bar_plot(mean_score_by_target, "Mean Score by Target", "Score", aggregate_dir / "mean_score_by_target.png")
    save_bar_plot(mean_loss_by_target, "Mean Loss by Target", "Loss", aggregate_dir / "mean_loss_by_target.png")

    score_arr = np.array([float(r["score"]) for r in records], dtype=np.float64)
    loss_arr = np.array([float(r["loss"]) for r in records], dtype=np.float64)
    selected_loss_arr = np.array([float(r["selected_loss"]) for r in records], dtype=np.float64)
    dc_arr = np.array([float(r["DC"]) for r in records], dtype=np.float64)
    hd_arr = np.array([float(r["HD"]) for r in records], dtype=np.float64)
    contact_count_arr = np.array([float(r["contact_count"]) for r in records], dtype=np.float64)
    active_count_arr = np.array([float(r["active_count"]) for r in records], dtype=np.float64)
    active_ratio_arr = np.array([float(r["active_ratio"]) for r in records], dtype=np.float64)
    wall_time_arr = np.array([float(r["wall_clock_time"]) for r in records], dtype=np.float64)
    greedy_time_arr = np.array([float(r["greedy_elapsed_seconds"]) for r in records], dtype=np.float64)
    norm_wall_arr = np.array([float(r["normalized_wall_clock_time"]) for r in records], dtype=np.float64)
    norm_greedy_arr = np.array([float(r["normalized_greedy_elapsed_time"]) for r in records], dtype=np.float64)
    evals_per_sec_arr = np.array([float(r["evals_per_second"]) for r in records], dtype=np.float64)
    model_time_per_call_arr = np.array([float(r["model_forward_time_per_call"]) for r in records], dtype=np.float64)

    save_scatter_plot(score_arr, loss_arr, "Score", "Loss", "Score vs Loss", aggregate_dir / "score_vs_loss.png")
    save_scatter_plot(score_arr, selected_loss_arr, "Score", "Selected Loss", "Score vs Selected Loss", aggregate_dir / "score_vs_selected_loss.png")
    save_scatter_plot(dc_arr, hd_arr, "DC", "HD", "DC vs HD", aggregate_dir / "dc_vs_hd.png")
    save_hist_plot(contact_count_arr, "Contact Count Distribution", "Contact Count", aggregate_dir / "contact_count_hist.png")
    save_hist_plot(active_count_arr, "Active Count Distribution", "Active Count", aggregate_dir / "active_count_hist.png")
    save_hist_plot(active_ratio_arr, "Active Ratio Distribution", "Active Ratio", aggregate_dir / "active_ratio_hist.png")
    save_hist_plot(wall_time_arr, "Wall Clock Time Distribution", "Seconds", aggregate_dir / "wall_clock_time_hist.png")
    save_hist_plot(greedy_time_arr, "Greedy Selection Time Distribution", "Seconds", aggregate_dir / "greedy_elapsed_hist.png")
    save_hist_plot(norm_wall_arr, "Normalized Wall Time Distribution", "Wall/Baseline", aggregate_dir / "normalized_wall_time_hist.png")
    save_hist_plot(norm_greedy_arr, "Normalized Greedy Time Distribution", "Greedy/Baseline", aggregate_dir / "normalized_greedy_time_hist.png")
    save_hist_plot(evals_per_sec_arr, "Evaluations per Second", "eval/sec", aggregate_dir / "evals_per_second_hist.png")
    save_hist_plot(model_time_per_call_arr, "Model Forward Time per Call", "seconds/call", aggregate_dir / "model_time_per_call_hist.png")


def run() -> None:
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    set_reproducibility(RESEARCH_SEED)

    subjects = SUBJECTS
    print(f"output dir: {output_root}")
    print(f"overwrite: {OVERWRITE}")
    print(f"cpu cores: total={TOTAL_CORES}, using={N_JOBS}")
    print(f"number of subjects: {len(subjects)}")
    target_paths = sorted(TARGET_DIR.glob("*.npy"))
    if not target_paths:
        raise FileNotFoundError(f"no target npy files found in: {TARGET_DIR}")
    print(f"number of targets: {len(target_paths)}")

    first_target = normalize_density(np.load(target_paths[0]))
    baseline_seconds = compute_baseline_seconds(first_target.shape)
    runtime_env = collect_runtime_environment()
    print(f"baseline_seconds: {baseline_seconds:.6f}")

    aggregate_records: list[dict] = []

    for (a, b, c), loss_name in zip(LOSS_COMB, LOSS_NAMES):
        for target_path in target_paths:
            target_name = target_path.stem
            target_density = normalize_density(np.load(target_path))
            if target_density.ndim != 2:
                raise ValueError(f"target map must be 2D, got shape={target_density.shape} for {target_path}")
            target_map_shape = target_density.shape

            for subject in subjects:
                data_dir = DATA_ROOT
                if not data_dir.exists():
                    print(f"[skip] subject {subject}: data dir not found -> {data_dir}")
                    continue

                try:
                    polar_map = load_nifti_fdata(data_dir / FNAME_ANG)
                    ecc_map = load_nifti_fdata(data_dir / FNAME_ECC)
                    sigma_map = load_nifti_fdata(data_dir / FNAME_SIGMA)
                    aparc_roi = load_nifti_fdata(data_dir / FNAME_APARC)
                    label_map = load_nifti_fdata(data_dir / FNAME_LABEL)
                except FileNotFoundError as exc:
                    print(f"[skip] subject {subject}: missing required file - {exc}")
                    continue

                dot = ecc_map * polar_map
                good_coords = np.asarray(np.where(dot != 0.0))

                cs_coords_rh = np.where(aparc_roi == 1021)
                cs_coords_lh = np.where(aparc_roi == 2021)
                gm_coords_rh = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
                gm_coords_lh = np.where(aparc_roi > 2000)

                xl, yl, zl = get_xyz(gm_coords_lh)
                xr, yr, zr = get_xyz(gm_coords_rh)
                gm_lh = np.array([xl, yl, zl]).T
                gm_rh = np.array([xr, yr, zr]).T

                v1_coords = np.asarray(np.where(label_map == 1))
                v2_coords = np.asarray(np.where(label_map == 2))
                v3_coords = np.asarray(np.where(label_map == 3))

                good_coords_lh = coords_intersection(good_coords, np.asarray(gm_coords_lh))
                good_coords_rh = coords_intersection(good_coords, np.asarray(gm_coords_rh))
                v1_coords_lh = coords_intersection(v1_coords, np.asarray(gm_coords_lh))
                v1_coords_rh = coords_intersection(v1_coords, np.asarray(gm_coords_rh))
                v2_coords_lh = coords_intersection(v2_coords, np.asarray(gm_coords_lh))
                v2_coords_rh = coords_intersection(v2_coords, np.asarray(gm_coords_rh))
                v3_coords_lh = coords_intersection(v3_coords, np.asarray(gm_coords_lh))
                v3_coords_rh = coords_intersection(v3_coords, np.asarray(gm_coords_rh))

                median_lh = [float(np.median(cs_coords_lh[0])), float(np.median(cs_coords_lh[1])), float(np.median(cs_coords_lh[2]))]
                median_rh = [float(np.median(cs_coords_rh[0])), float(np.median(cs_coords_rh[1])), float(np.median(cs_coords_rh[2]))]

                print(f"target: {target_name}")
                print(f"loss: {loss_name}")
                print(f"a,b,c: {a},{b},{c}")

                hemi_iter: Iterable[tuple[np.ndarray, str, Sequence[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Integer]] = [
                    (gm_lh, "LH", median_lh, good_coords_lh, v1_coords_lh, v2_coords_lh, v3_coords_lh, DIM_BETA_LH),
                    (gm_rh, "RH", median_rh, good_coords_rh, v1_coords_rh, v2_coords_rh, v3_coords_rh, DIM_BETA_RH),
                ]

                for gm_mask, hem, start_location, good_h, v1_h, v2_h, v3_h, dim_beta in hemi_iter:
                    data_id = f"{subject}_{hem}_V1_n1000_1x10_{loss_name}_{THRESH}_{target_name}"
                    run_dir = output_root / target_name / data_id
                    results_json_path = run_dir / "results.json"

                    if results_json_path.exists() and not OVERWRITE:
                        print(f"[skip] {run_dir.name} already exists")
                        existing_record = load_aggregate_record_from_results(
                            results_json_path,
                            run_dir,
                            fallback_target_name=target_name,
                            fallback_subject=subject,
                            fallback_hemisphere=hem,
                        )
                        if existing_record is not None:
                            aggregate_records.append(existing_record)
                        continue

                    run_dir.mkdir(parents=True, exist_ok=True)
                    wall_t0 = perf_counter()

                    dimensions = [DIM_ALPHA, dim_beta, DIM_OFFSET, DIM_SHANK]
                    lhs2 = "lhs"

                    timing_stats = {
                        "model_forward_seconds": 0.0,
                        "simulator_forward_seconds": 0.0,
                        "metric_computation_seconds": 0.0,
                    }
                    call_stats = {
                        "model_forward_calls": 0,
                        "simulator_forward_calls": 0,
                    }

                    @use_named_args(dimensions=dimensions)
                    def objective(alpha, beta, offset_from_base, shank_length):
                        warnings.filterwarnings(
                            "ignore",
                            message=r"Signature .* for <class 'numpy\.longdouble'> does not match any known type.*",
                            category=UserWarning,
                        )
                        penalty = 0.25
                        new_angle = (float(alpha), float(beta), 0)

                        orig_grid = create_grid(
                            start_location,
                            shank_length=shank_length,
                            n_contactpoints_shank=N_CONTACTPOINTS_SHANK,
                            spacing_along_xy=SPACING_ALONG_XY,
                            offset_from_origin=0,
                        )

                        _, contacts_xyz_moved, _, _, _, _, _, _, grid_valid = implant_grid(
                            gm_mask, orig_grid, start_location, new_angle, offset_from_base
                        )

                        model_t0 = perf_counter()
                        phos_v1 = get_phosphenes(contacts_xyz_moved, v1_h, polar_map, ecc_map, sigma_map)
                        timing_stats["model_forward_seconds"] += perf_counter() - model_t0
                        call_stats["model_forward_calls"] += 1

                        if phos_v1.size == 0:
                            return 3.0

                        m_inv = 1 / get_cortical_magnification(phos_v1[:, 1], CORT_MAG_MODEL)
                        spread = cortical_spread(AMP)
                        sigmas = (spread * m_inv) / 2
                        phos_v1[:, 2] = sigmas

                        phosphene_map = np.zeros(target_map_shape, dtype="float32")
                        sim_t0 = perf_counter()
                        phosphene_map = prf_to_phos(phosphene_map, phos_v1, view_angle=VIEW_ANGLE, phSizeScale=1)
                        timing_stats["simulator_forward_seconds"] += perf_counter() - sim_t0
                        call_stats["simulator_forward_calls"] += 1

                        max_ph = np.max(phosphene_map)
                        sum_ph = np.sum(phosphene_map)
                        if max_ph > 0:
                            phosphene_map /= max_ph
                        if sum_ph > 0:
                            phosphene_map /= np.sum(phosphene_map)

                        metric_t0 = perf_counter()
                        dice, _, _ = DC(target_density, phosphene_map, np.percentile(target_density, DC_PERCENTILE))
                        grid_yield = get_yield(contacts_xyz_moved, good_h)
                        hell_d = hellinger_distance(phosphene_map.flatten(), target_density.flatten())
                        timing_stats["metric_computation_seconds"] += perf_counter() - metric_t0

                        par1 = 1.0 - (a * dice)
                        par2 = 1.0 - (b * grid_yield)
                        par3 = 1 if (np.isnan(hell_d) or np.isinf(hell_d)) else (c * hell_d)
                        cost = par1 + par2 + par3

                        if np.isnan(phosphene_map).any() or np.sum(phosphene_map) == 0:
                            cost = 3.0
                        if not grid_valid:
                            cost += penalty * 3
                        if np.isnan(cost) or np.isinf(cost):
                            cost = 3.0

                        return float(cost)

                    def _callback_with_tqdm(res):
                        pbar.update(1)
                        return custom_stopper(res)

                    pbar = tqdm(
                        total=NUM_CALLS,
                        desc=f"{subject}_{hem}_{target_name}",
                        unit="eval",
                        leave=True,
                    )
                    optimization_t0 = perf_counter()
                    try:
                        res: Any = gp_minimize(
                            objective,
                            x0=X0,
                            dimensions=dimensions,
                            n_jobs=N_JOBS,
                            n_calls=NUM_CALLS,
                            n_initial_points=NUM_INITIAL_POINTS,
                            initial_point_generator=lhs2,
                            callback=[_callback_with_tqdm],
                        )
                    finally:
                        pbar.close()
                    assert res is not None
                    optimization_elapsed_seconds = perf_counter() - optimization_t0

                    print(
                        f"subject {subject} {hem}, best alpha: {res.x[0]}, "
                        f"best beta: {res.x[1]}, best offset_from_base: {res.x[2]}, best shank_length: {res.x[3]}"
                    )

                    recompute_t0 = perf_counter()
                    print(f"[{subject} {hem} {target_name}] recomputing best solution...", flush=True)
                    best_alpha, best_beta, best_offset, best_shank = res.x
                    new_angle = (float(best_alpha), float(best_beta), 0)
                    orig_grid = create_grid(
                        start_location,
                        shank_length=best_shank,
                        n_contactpoints_shank=N_CONTACTPOINTS_SHANK,
                        spacing_along_xy=SPACING_ALONG_XY,
                        offset_from_origin=0,
                    )
                    _, contacts_xyz_moved, _, _, _, _, _, _, grid_valid = implant_grid(
                        gm_mask, orig_grid, start_location, new_angle, best_offset
                    )

                    model_t0 = perf_counter()
                    _ = get_phosphenes(contacts_xyz_moved, good_h, polar_map, ecc_map, sigma_map)
                    phos_v1 = get_phosphenes(contacts_xyz_moved, v1_h, polar_map, ecc_map, sigma_map)
                    _ = get_phosphenes(contacts_xyz_moved, v2_h, polar_map, ecc_map, sigma_map)
                    _ = get_phosphenes(contacts_xyz_moved, v3_h, polar_map, ecc_map, sigma_map)
                    timing_stats["model_forward_seconds"] += perf_counter() - model_t0
                    call_stats["model_forward_calls"] += 4

                    if phos_v1.size > 0:
                        m_inv = 1 / get_cortical_magnification(phos_v1[:, 1], CORT_MAG_MODEL)
                        spread = cortical_spread(AMP)
                        phos_v1[:, 2] = (spread * m_inv) / 2

                    phosphene_map = np.zeros(target_map_shape, dtype="float32")
                    sim_t0 = perf_counter()
                    phosphene_map = prf_to_phos(phosphene_map, phos_v1, view_angle=VIEW_ANGLE, phSizeScale=1)
                    timing_stats["simulator_forward_seconds"] += perf_counter() - sim_t0
                    call_stats["simulator_forward_calls"] += 1

                    if np.max(phosphene_map) > 0:
                        phosphene_map /= np.max(phosphene_map)
                    if np.sum(phosphene_map) > 0:
                        phosphene_map /= np.sum(phosphene_map)

                    metric_t0 = perf_counter()
                    dice, _, _ = DC(target_density, phosphene_map, np.percentile(target_density, DC_PERCENTILE))
                    grid_yield = get_yield(contacts_xyz_moved, good_h)
                    hell_d = hellinger_distance(phosphene_map.flatten(), target_density.flatten())
                    timing_stats["metric_computation_seconds"] += perf_counter() - metric_t0
                    recompute_elapsed_seconds = perf_counter() - recompute_t0

                    score = float(dice + (0.1 * grid_yield) - hell_d)
                    loss = float(2.0 - score)

                    print(f"[{subject} {hem} {target_name}] starting greedy selection...", flush=True)
                    greedy_t0 = perf_counter()
                    active_idx, selected_map, loss_history, progress = greedy_select(
                        phosphenes_v1=phos_v1,
                        contacts_xyz_moved=contacts_xyz_moved,
                        good_coords=good_h,
                        grid_valid=grid_valid,
                        target_density=target_density,
                        target_shape=target_map_shape,
                        n_jobs=N_JOBS,
                    )
                    greedy_elapsed_seconds = perf_counter() - greedy_t0

                    selected_contacts = subset_contacts(contacts_xyz_moved, active_idx)
                    selected_loss, selected_comps = compute_loss(
                        target_density,
                        selected_map,
                        selected_contacts,
                        good_h,
                        grid_valid,
                    )

                    contact_count = int(phos_v1.shape[0])
                    active_count = int(len(active_idx))
                    active_ratio = float(active_count / max(1, contact_count))

                    full_loss = float(loss)
                    full_dice = float(dice)
                    full_hell_d = float(hell_d)
                    full_grid_yield = float(grid_yield)
                    total_elapsed_seconds = float(perf_counter() - wall_t0)

                    n_evals = max(1, int(len(res.x_iters)))
                    wall_clock_time = total_elapsed_seconds
                    model_forward_time = float(timing_stats["model_forward_seconds"])
                    simulator_forward_time = float(timing_stats["simulator_forward_seconds"])
                    metric_computation_time = float(timing_stats["metric_computation_seconds"])

                    model_forward_calls = max(1, int(call_stats["model_forward_calls"]))
                    simulator_forward_calls = max(1, int(call_stats["simulator_forward_calls"]))
                    normalized_wall = float(wall_clock_time / baseline_seconds) if baseline_seconds > 0 else np.nan
                    normalized_model = float(model_forward_time / baseline_seconds) if baseline_seconds > 0 else np.nan
                    normalized_sim = float(simulator_forward_time / baseline_seconds) if baseline_seconds > 0 else np.nan
                    normalized_metric = float(metric_computation_time / baseline_seconds) if baseline_seconds > 0 else np.nan
                    normalized_greedy = float(greedy_elapsed_seconds / baseline_seconds) if baseline_seconds > 0 else np.nan

                    model_forward_time_per_call = float(model_forward_time / model_forward_calls)
                    simulator_forward_time_per_call = float(simulator_forward_time / simulator_forward_calls)
                    metric_time_per_eval = float(metric_computation_time / n_evals)
                    wall_clock_time_per_eval = float(wall_clock_time / n_evals)
                    evals_per_second = float(n_evals / max(wall_clock_time, 1e-12))
                    model_calls_per_eval = float(call_stats["model_forward_calls"] / n_evals)
                    simulator_calls_per_eval = float(call_stats["simulator_forward_calls"] / n_evals)

                    electrode_activations = compute_electrode_activations(contacts_xyz_moved, good_h)
                    good_contact_count = int(np.count_nonzero(electrode_activations > 0))
                    good_contact_ratio = float(good_contact_count / max(1, contact_count))
                    active_contact_count = int(active_count)
                    mean_activation = float(np.mean(electrode_activations)) if electrode_activations.size > 0 else 0.0
                    max_activation = float(np.max(electrode_activations)) if electrode_activations.size > 0 else 0.0
                    min_activation = float(np.min(electrode_activations)) if electrode_activations.size > 0 else 0.0

                    print(f"best dice, yield, HD: {dice}, {grid_yield}, {hell_d}")
                    print(f"greedy selected contacts: {active_count}/{contact_count}")

                    write_t0 = perf_counter()
                    print(f"[{subject} {hem} {target_name}] writing artifacts...", flush=True)
                    np.save(run_dir / "electrode_activations.npy", electrode_activations.astype(np.float32))
                    np.save(run_dir / "greedy_selected_indices.npy", np.asarray(active_idx, dtype=np.int32))
                    np.save(run_dir / "greedy_full_map.npy", np.asarray(phosphene_map, dtype=np.float32))
                    np.save(run_dir / "greedy_selected_map.npy", np.asarray(selected_map, dtype=np.float32))
                    np.save(run_dir / "greedy_loss_history.npy", np.asarray(loss_history, dtype=np.float64))
                    np.save(run_dir / "greedy_progress_active_count.npy", np.asarray([p["active_count"] for p in progress], dtype=np.float64))
                    np.save(run_dir / "greedy_progress_loss.npy", np.asarray([p["loss"] for p in progress], dtype=np.float64))
                    np.save(run_dir / "greedy_progress_dice.npy", np.asarray([p["dice"] for p in progress], dtype=np.float64))
                    np.save(run_dir / "greedy_progress_hell.npy", np.asarray([p["hell"] for p in progress], dtype=np.float64))
                    np.save(run_dir / "greedy_progress_yield.npy", np.asarray([p["yield"] for p in progress], dtype=np.float64))
                    np.save(run_dir / "reconstruction.npy", phosphene_map.astype(np.float32))
                    np.save(run_dir / "selected_reconstruction.npy", np.asarray(selected_map, dtype=np.float32))
                    np.save(run_dir / "target.npy", target_density.astype(np.float32))

                    save_density_plot(target_density, "Target Density", run_dir / "target.png")
                    save_density_plot(phosphene_map, "Reconstruction", run_dir / "reconstruction.png")
                    save_density_plot(selected_map, "Greedy Selected Reconstruction", run_dir / "selected_reconstruction.png")
                    save_comparison_plot(target_density, phosphene_map, run_dir / "comparison.png")
                    artifact_write_time = float(perf_counter() - write_t0)
                    print(f"[{subject} {hem} {target_name}] finished writing artifacts", flush=True)

                    results = {
                        "subject": subject,
                        "hemisphere": hem,
                        "target": {
                            "path": str(target_path),
                            "name": target_name,
                            "shape": [int(target_map_shape[0]), int(target_map_shape[1])],
                        },
                        "optimization": {
                            "loss_name": loss_name,
                            "weights": {"a": float(a), "b": float(b), "c": float(c)},
                            "best_params": {
                                "alpha": int(best_alpha),
                                "beta": int(best_beta),
                                "offset_from_base": int(best_offset),
                                "shank_length": int(best_shank),
                            },
                            "best_optimizer_fun": float(res.fun),
                            "n_calls": int(len(res.x_iters)),
                            "seed": int(RESEARCH_SEED),
                        },
                        "reconstruction_metrics": {
                            "DC": full_dice,
                            "Y": full_grid_yield,
                            "HD": full_hell_d,
                        },
                        "composite_scores": {
                            "score": score,
                            "loss": loss,
                            "formula": "Score = DC + 0.1*Y - HD, Loss = 2 - Score",
                        },
                        "efficiency_metrics": {
                            "contact_count": contact_count,
                            "active_count": active_count,
                            "active_contact_count": active_contact_count,
                            "active_ratio": active_ratio,
                            "good_contact_count": good_contact_count,
                            "good_contact_ratio": good_contact_ratio,
                            "mean_activation": mean_activation,
                            "max_activation": max_activation,
                            "min_activation": min_activation,
                        },
                        "greedy_metrics": {
                            "contact_count": contact_count,
                            "active_count": active_count,
                            "active_ratio": active_ratio,
                            "selected_loss": float(selected_loss),
                            "selected_dice": float(selected_comps["dice"]),
                            "selected_hell_d": float(selected_comps["hell"]),
                            "selected_grid_yield": float(selected_comps["yield"]),
                            "original_loss": full_loss,
                            "original_dice": full_dice,
                            "original_hell_d": full_hell_d,
                            "original_grid_yield": full_grid_yield,
                            "greedy_elapsed_seconds": float(greedy_elapsed_seconds),
                        },
                        "simulator_metrics": {
                            "simulator_forward_calls": int(call_stats["simulator_forward_calls"]),
                            "model_forward_calls": int(call_stats["model_forward_calls"]),
                            "simulator_calls_per_eval": simulator_calls_per_eval,
                            "model_calls_per_eval": model_calls_per_eval,
                        },
                        "timing_seconds": {
                            "wall_clock_time": wall_clock_time,
                            "optimization_time": float(optimization_elapsed_seconds),
                            "recompute_time": float(recompute_elapsed_seconds),
                            "artifact_write_time": artifact_write_time,
                            "model_forward_time": model_forward_time,
                            "simulator_forward_time": simulator_forward_time,
                            "metric_computation_time": metric_computation_time,
                            "greedy_elapsed_time": float(greedy_elapsed_seconds),
                        },
                        "timing_normalized": {
                            "baseline_seconds": float(baseline_seconds),
                            "baseline_repeats": int(BASELINE_REPEATS),
                            "normalized_wall_clock_time": normalized_wall,
                            "normalized_model_forward_time": normalized_model,
                            "normalized_simulator_forward_time": normalized_sim,
                            "normalized_metric_computation_time": normalized_metric,
                            "normalized_greedy_elapsed_time": float(greedy_elapsed_seconds / baseline_seconds) if baseline_seconds > 0 else np.nan,
                            "wall_clock_time_per_eval": wall_clock_time_per_eval,
                            "metric_time_per_eval": metric_time_per_eval,
                            "model_forward_time_per_call": model_forward_time_per_call,
                            "simulator_forward_time_per_call": simulator_forward_time_per_call,
                            "evals_per_second": evals_per_second,
                        },
                        "runtime_environment": runtime_env,
                        "artifacts": {
                            "results_json": "results.json",
                            "electrode_activations_npy": "electrode_activations.npy",
                            "reconstruction_npy": "reconstruction.npy",
                            "selected_reconstruction_npy": "selected_reconstruction.npy",
                            "greedy_selected_indices_npy": "greedy_selected_indices.npy",
                            "greedy_full_map_npy": "greedy_full_map.npy",
                            "greedy_selected_map_npy": "greedy_selected_map.npy",
                            "greedy_loss_history_npy": "greedy_loss_history.npy",
                            "target_npy": "target.npy",
                            "reconstruction_png": "reconstruction.png",
                            "selected_reconstruction_png": "selected_reconstruction.png",
                            "target_png": "target.png",
                            "comparison_png": "comparison.png",
                        },
                        "grid_valid": bool(grid_valid),
                    }

                    with open(results_json_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
                    print(f"saved: {run_dir}")

                    aggregate_records.append(
                        {
                            "target_name": target_name,
                            "subject": subject,
                            "hemisphere": hem,
                            "run_dir": str(run_dir),
                            "DC": float(full_dice),
                            "Y": float(full_grid_yield),
                            "HD": float(full_hell_d),
                            "score": float(score),
                            "loss": float(loss),
                            "contact_count": int(contact_count),
                            "active_count": int(active_count),
                            "active_ratio": float(active_ratio),
                            "selected_loss": float(selected_loss),
                            "selected_dice": float(selected_comps["dice"]),
                            "selected_hell_d": float(selected_comps["hell"]),
                            "selected_grid_yield": float(selected_comps["yield"]),
                            "greedy_elapsed_seconds": float(greedy_elapsed_seconds),
                            "simulator_forward_calls": int(call_stats["simulator_forward_calls"]),
                            "model_forward_calls": int(call_stats["model_forward_calls"]),
                            "simulator_calls_per_eval": float(simulator_calls_per_eval),
                            "model_calls_per_eval": float(model_calls_per_eval),
                            "wall_clock_time": float(wall_clock_time),
                            "model_forward_time": float(model_forward_time),
                            "simulator_forward_time": float(simulator_forward_time),
                            "metric_computation_time": float(metric_computation_time),
                            "greedy_elapsed_time": float(greedy_elapsed_seconds),
                            "normalized_wall_clock_time": float(normalized_wall),
                            "normalized_greedy_elapsed_time": float(normalized_greedy),
                            "evals_per_second": float(evals_per_second),
                            "model_forward_time_per_call": float(model_forward_time_per_call),
                            "seed": int(RESEARCH_SEED),
                            "grid_valid": bool(grid_valid),
                        }
                    )

    write_aggregate_outputs(aggregate_records, output_root / "aggregate")
    print(f"aggregate saved: {output_root / 'aggregate'}")


if __name__ == "__main__":
    run()
