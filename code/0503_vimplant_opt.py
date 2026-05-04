# D:\yongtae\vimplant\code\0421_vimplant_opt.py

from __future__ import annotations

import json
import csv
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
from skopt.utils import cook_initial_point_generator, use_named_args

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

DC: Any = _DC
from electphos import (
    cortical_spread,
    create_grid,
    get_cortical_magnification,
    get_phosphenes,
    implant_grid,
    prf_to_phos,
)


if not np.geterr().get("divide"):
    warnings.simplefilter("ignore")
np.seterr(divide="ignore", invalid="ignore")


# Run configuration
OUTPUT_ROOT = PROJECT_ROOT / "data" / "0503_output"
TOTAL_CORES = os.cpu_count() or 1
N_JOBS = max(1, TOTAL_CORES)
OVERWRITE = False
RESEARCH_SEED = 42
BASELINE_REPEATS = 5

DATASET_DIRS: list[tuple[str, Path]] = []
for _ds in ("hcp_fmri", "deep_fmri"):
    _ds_path = PROJECT_ROOT / "data" / _ds
    if _ds_path.exists():
        for _sub_path in sorted(_ds_path.iterdir()):
            if _sub_path.is_dir():
                DATASET_DIRS.append((_ds, _sub_path))

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

LOSS_COMB = [(1, 0.1, 1)]
LOSS_NAMES = ["dice-yield-HD"]
THRESH = 0.05

TARGET_DIR = PROJECT_ROOT / "data" / "targets0503"
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
        "mean_good_contact_count": float(np.mean([float(r["good_contact_count"]) for r in records])),
        "std_good_contact_count": float(np.std([float(r["good_contact_count"]) for r in records])),
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
    dc_arr = np.array([float(r["DC"]) for r in records], dtype=np.float64)
    hd_arr = np.array([float(r["HD"]) for r in records], dtype=np.float64)
    active_ratio_arr = np.array([float(r["active_ratio"]) for r in records], dtype=np.float64)
    wall_time_arr = np.array([float(r["wall_clock_time"]) for r in records], dtype=np.float64)
    norm_wall_arr = np.array([float(r["normalized_wall_clock_time"]) for r in records], dtype=np.float64)
    evals_per_sec_arr = np.array([float(r["evals_per_second"]) for r in records], dtype=np.float64)
    model_time_per_call_arr = np.array([float(r["model_forward_time_per_call"]) for r in records], dtype=np.float64)

    save_scatter_plot(score_arr, loss_arr, "Score", "Loss", "Score vs Loss", aggregate_dir / "score_vs_loss.png")
    save_scatter_plot(dc_arr, hd_arr, "DC", "HD", "DC vs HD", aggregate_dir / "dc_vs_hd.png")
    save_hist_plot(active_ratio_arr, "Active Ratio Distribution", "Active Ratio", aggregate_dir / "active_ratio_hist.png")
    save_hist_plot(wall_time_arr, "Wall Clock Time Distribution", "Seconds", aggregate_dir / "wall_clock_time_hist.png")
    save_hist_plot(norm_wall_arr, "Normalized Wall Time Distribution", "Wall/Baseline", aggregate_dir / "normalized_wall_time_hist.png")
    save_hist_plot(evals_per_sec_arr, "Evaluations per Second", "eval/sec", aggregate_dir / "evals_per_second_hist.png")
    save_hist_plot(model_time_per_call_arr, "Model Forward Time per Call", "seconds/call", aggregate_dir / "model_time_per_call_hist.png")


def run() -> None:
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    set_reproducibility(RESEARCH_SEED)

    if not DATASET_DIRS:
        raise FileNotFoundError("no samples found in hcp_fmri or deep_fmri directories")
    print(f"output dir: {output_root}")
    print(f"overwrite: {OVERWRITE}")
    print(f"cpu cores: total={TOTAL_CORES}, using={N_JOBS}")
    print(f"number of samples: {len(DATASET_DIRS)}")
    target_paths = sorted(TARGET_DIR.glob("*.npy"))
    if not target_paths:
        raise FileNotFoundError(f"no target npy files found in: {TARGET_DIR}")
    print(f"number of targets: {len(target_paths)}")

    first_target = normalize_density(np.load(target_paths[0]))
    baseline_seconds = compute_baseline_seconds(first_target.shape)
    runtime_env = collect_runtime_environment()
    print(f"baseline_seconds: {baseline_seconds:.6f}")

    global_aggregate_records: list[dict] = []

    for dataset_name, data_dir in DATASET_DIRS:
        subject = data_dir.name
        sample_id = f"{dataset_name}_{subject}"
        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== sample: {sample_id} ===")

        if not data_dir.exists():
            print(f"[skip] {sample_id}: data dir not found -> {data_dir}")
            continue

        try:
            polar_map = cast(Any, nib).load(str(data_dir / FNAME_ANG)).get_fdata()
            ecc_map = cast(Any, nib).load(str(data_dir / FNAME_ECC)).get_fdata()
            sigma_map = cast(Any, nib).load(str(data_dir / FNAME_SIGMA)).get_fdata()
            aparc_roi = cast(Any, nib).load(str(data_dir / FNAME_APARC)).get_fdata()
            label_map = cast(Any, nib).load(str(data_dir / FNAME_LABEL)).get_fdata()
        except FileNotFoundError as exc:
            print(f"[skip] {sample_id}: missing required file - {exc}")
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

        median_lh = [np.median(cs_coords_lh[0]), np.median(cs_coords_lh[1]), np.median(cs_coords_lh[2])]
        median_rh = [np.median(cs_coords_rh[0]), np.median(cs_coords_rh[1]), np.median(cs_coords_rh[2])]

        sample_aggregate_records: list[dict] = []

        for (a, b, c), loss_name in zip(LOSS_COMB, LOSS_NAMES):
            for target_path in target_paths:
                target_name = target_path.stem
                target_density = normalize_density(np.load(target_path))
                if target_density.ndim != 2:
                    raise ValueError(f"target map must be 2D, got shape={target_density.shape} for {target_path}")
                target_map_shape = target_density.shape

                print(f"target: {target_name}")
                print(f"loss: {loss_name}")
                print(f"a,b,c: {a},{b},{c}")

                hemi_iter: Iterable[tuple[np.ndarray, str, list[Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Integer]] = [
                    (gm_lh, "LH", median_lh, good_coords_lh, v1_coords_lh, v2_coords_lh, v3_coords_lh, DIM_BETA_LH),
                    (gm_rh, "RH", median_rh, good_coords_rh, v1_coords_rh, v2_coords_rh, v3_coords_rh, DIM_BETA_RH),
                ]

                for gm_mask, hem, start_location, good_h, v1_h, v2_h, v3_h, dim_beta in hemi_iter:
                    data_id = f"{dataset_name}_{subject}_{hem}_V1_n1000_1x10_{loss_name}_{THRESH}_{target_name}"
                    run_dir = sample_dir / target_name / data_id
                    results_json_path = run_dir / "results.json"

                    if results_json_path.exists() and not OVERWRITE:
                        print(f"[skip] {run_dir.name} already exists")
                        continue

                    run_dir.mkdir(parents=True, exist_ok=True)
                    wall_t0 = perf_counter()

                    dimensions = [DIM_ALPHA, dim_beta, DIM_OFFSET, DIM_SHANK]
                    lhs2: Any = cook_initial_point_generator("lhs", criterion="maximin")

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
                        res = gp_minimize(
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
                    optimization_elapsed_seconds = perf_counter() - optimization_t0

                    if res is None:
                        continue
                    print(
                        f"subject {subject} {hem}, best alpha: {res.x[0]}, "
                        f"best beta: {res.x[1]}, best offset_from_base: {res.x[2]}, best shank_length: {res.x[3]}"
                    )

                    recompute_t0 = perf_counter()
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

                    n_evals = max(1, int(len(res.x_iters))) if res.x_iters is not None else 1
                    wall_clock_time = float(perf_counter() - wall_t0)
                    model_forward_time = float(timing_stats["model_forward_seconds"])
                    simulator_forward_time = float(timing_stats["simulator_forward_seconds"])
                    metric_computation_time = float(timing_stats["metric_computation_seconds"])

                    model_forward_calls = max(1, int(call_stats["model_forward_calls"]))
                    simulator_forward_calls = max(1, int(call_stats["simulator_forward_calls"]))
                    normalized_wall = float(wall_clock_time / baseline_seconds) if baseline_seconds > 0 else np.nan
                    normalized_model = float(model_forward_time / baseline_seconds) if baseline_seconds > 0 else np.nan
                    normalized_sim = float(simulator_forward_time / baseline_seconds) if baseline_seconds > 0 else np.nan
                    normalized_metric = float(metric_computation_time / baseline_seconds) if baseline_seconds > 0 else np.nan

                    model_forward_time_per_call = float(model_forward_time / model_forward_calls)
                    simulator_forward_time_per_call = float(simulator_forward_time / simulator_forward_calls)
                    metric_time_per_eval = float(metric_computation_time / n_evals)
                    wall_clock_time_per_eval = float(wall_clock_time / n_evals)
                    evals_per_second = float(n_evals / max(wall_clock_time, 1e-12))
                    model_calls_per_eval = float(call_stats["model_forward_calls"] / n_evals)
                    simulator_calls_per_eval = float(call_stats["simulator_forward_calls"] / n_evals)

                    electrode_activations = compute_electrode_activations(contacts_xyz_moved, good_h)
                    contact_count = int(electrode_activations.size)
                    active_count = int(contact_count)
                    active_contact_count = int(active_count)
                    active_ratio = float(active_count / max(1, contact_count))
                    good_contact_count = int(np.count_nonzero(electrode_activations > 0))
                    good_contact_ratio = float(good_contact_count / max(1, contact_count))
                    mean_activation = float(np.mean(electrode_activations)) if electrode_activations.size > 0 else 0.0
                    max_activation = float(np.max(electrode_activations)) if electrode_activations.size > 0 else 0.0
                    min_activation = float(np.min(electrode_activations)) if electrode_activations.size > 0 else 0.0

                    print(f"best dice, yield, HD: {dice}, {grid_yield}, {hell_d}")

                    write_t0 = perf_counter()
                    np.save(run_dir / "electrode_activations.npy", electrode_activations.astype(np.float32))
                    np.save(run_dir / "reconstruction.npy", phosphene_map.astype(np.float32))
                    np.save(run_dir / "target.npy", target_density.astype(np.float32))

                    save_density_plot(target_density, "Target Density", run_dir / "target.png")
                    save_density_plot(phosphene_map, "Reconstruction", run_dir / "reconstruction.png")
                    save_comparison_plot(target_density, phosphene_map, run_dir / "comparison.png")
                    artifact_write_time = float(perf_counter() - write_t0)

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
                            "best_optimizer_fun": float(res.fun) if res.fun is not None else 0.0,
                            "n_calls": int(len(res.x_iters)) if res.x_iters is not None else 0,
                            "seed": int(RESEARCH_SEED),
                        },
                        "reconstruction_metrics": {
                            "DC": float(dice),
                            "Y": float(grid_yield),
                            "HD": float(hell_d),
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
                            "selected_loss": float(loss),
                            "selected_dice": float(dice),
                            "selected_hell_d": float(hell_d),
                            "selected_grid_yield": float(grid_yield),
                            "original_loss": float(loss),
                            "original_dice": float(dice),
                            "original_hell_d": float(hell_d),
                            "original_grid_yield": float(grid_yield),
                            "greedy_elapsed_seconds": 0.0,
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
                            "greedy_elapsed_time": 0.0,
                        },
                        "timing_normalized": {
                            "baseline_seconds": float(baseline_seconds),
                            "baseline_repeats": int(BASELINE_REPEATS),
                            "normalized_wall_clock_time": normalized_wall,
                            "normalized_model_forward_time": normalized_model,
                            "normalized_simulator_forward_time": normalized_sim,
                            "normalized_metric_computation_time": normalized_metric,
                            "normalized_greedy_elapsed_time": 0.0,
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
                            "selected_reconstruction_npy": None,
                            "greedy_selected_indices_npy": None,
                            "greedy_full_map_npy": None,
                            "greedy_selected_map_npy": None,
                            "greedy_loss_history_npy": None,
                            "target_npy": "target.npy",
                            "reconstruction_png": "reconstruction.png",
                            "selected_reconstruction_png": None,
                            "target_png": "target.png",
                            "comparison_png": "comparison.png",
                        },
                        "grid_valid": bool(grid_valid),
                    }

                    with open(results_json_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
                    print(f"saved: {run_dir}")

                    record = {
                        "dataset": dataset_name,
                        "target_name": target_name,
                        "subject": subject,
                        "hemisphere": hem,
                        "run_dir": str(run_dir),
                        "DC": float(dice),
                        "Y": float(grid_yield),
                        "HD": float(hell_d),
                        "score": float(score),
                        "loss": float(loss),
                        "contact_count": int(contact_count),
                        "active_count": int(active_count),
                        "active_contact_count": int(active_contact_count),
                        "active_ratio": float(active_ratio),
                        "good_contact_count": int(good_contact_count),
                        "good_contact_ratio": float(good_contact_ratio),
                        "mean_activation": float(mean_activation),
                        "max_activation": float(max_activation),
                        "min_activation": float(min_activation),
                        "simulator_forward_calls": int(call_stats["simulator_forward_calls"]),
                        "model_forward_calls": int(call_stats["model_forward_calls"]),
                        "simulator_calls_per_eval": float(simulator_calls_per_eval),
                        "model_calls_per_eval": float(model_calls_per_eval),
                        "wall_clock_time": float(wall_clock_time),
                        "model_forward_time": float(model_forward_time),
                        "simulator_forward_time": float(simulator_forward_time),
                        "metric_computation_time": float(metric_computation_time),
                        "normalized_wall_clock_time": float(normalized_wall),
                        "evals_per_second": float(evals_per_second),
                        "model_forward_time_per_call": float(model_forward_time_per_call),
                        "seed": int(RESEARCH_SEED),
                        "grid_valid": bool(grid_valid),
                    }
                    sample_aggregate_records.append(record)
                    global_aggregate_records.append(record)

        write_aggregate_outputs(sample_aggregate_records, sample_dir / "aggregate")
        print(f"sample aggregate saved: {sample_dir / 'aggregate'}")

    write_aggregate_outputs(global_aggregate_records, output_root / "aggregate")
    print(f"global aggregate saved: {output_root / 'aggregate'}")


if __name__ == "__main__":
    run()
