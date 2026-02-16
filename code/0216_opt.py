# code/0216_opt.py

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
from typing import Iterable, Sequence

# Hide known non-critical longdouble warning on some WSL/numpy builds.
warnings.filterwarnings(
    "ignore",
    message=r"Signature .* for <class 'numpy\.longdouble'> does not match any known type.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
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
from lossfunc import DC, get_yield, hellinger_distance
from electphos import (
    cortical_spread,
    create_grid,
    get_cortical_magnification,
    get_phosphenes,
    implant_grid,
    prf_to_phos,
)
import visualsectors as gvs


if not np.geterr().get("divide"):
    warnings.simplefilter("ignore")
np.seterr(divide="ignore", invalid="ignore")


# Hardcoded run configuration
SUBJECTS = ["100610"]
DATA_ROOT = PROJECT_ROOT / "data" / "input" / "100610"
OUTPUT_ROOT = Path("data/output/opt")

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
WINDOWSIZE = 1000

LOSS_COMB = [(1, 0.1, 1)]
LOSS_NAMES = ["dice-yield-HD"]
THRESH = 0.05

TARGET_MAPS = [
    gvs.upper_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False),
    gvs.lower_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False),
    gvs.inner_ring(windowsize=WINDOWSIZE, fwhm=400, radiusLow=0, radiusHigh=250, plotting=False),
    gvs.complete_gauss(windowsize=WINDOWSIZE, fwhm=1200, radiusLow=0, radiusHigh=500, center=None, plotting=False),
]
TARGET_NAMES = ["targ-upper", "targ-lower", "targ-inner", "targ-full"]

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


def save_result_text(
    txt_path: Path,
    subject: str,
    hemisphere: str,
    target_name: str,
    loss_name: str,
    best_x: Sequence[float],
    grid_valid: bool,
    dice: float,
    hell_d: float,
    grid_yield: float,
) -> None:
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"Hemisphere: {hemisphere}\n")
        f.write(f"Target: {target_name}\n")
        f.write(f"Loss: {loss_name}\n")
        f.write(f"Threshold: {THRESH}\n")
        f.write("Best parameters:\n")
        f.write(f"  Alpha: {best_x[0]}\n")
        f.write(f"  Beta: {best_x[1]}\n")
        f.write(f"  Offset from base: {best_x[2]}\n")
        f.write(f"  Shank length: {best_x[3]}\n")
        f.write("Results:\n")
        f.write(f"  Grid valid: {grid_valid}\n")
        f.write(f"  Dice coefficient: {dice:.6f}\n")
        f.write(f"  Hellinger distance: {hell_d:.6f}\n")
        f.write(f"  Grid yield: {grid_yield:.6f}\n")


def run() -> None:
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = SUBJECTS
    print(f"number of subjects: {len(subjects)}")

    for target_template, target_name in zip(TARGET_MAPS, TARGET_NAMES):
        target_density = normalize_density(target_template)
        bin_thresh = np.percentile(target_density, DC_PERCENTILE)
        target_density_bin = (target_density > bin_thresh).astype(bool)
        fig, _ = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplot(1, 2, 1).imshow(target_density, cmap="seismic")
        plt.title("Target density")
        plt.subplot(1, 2, 2).imshow(target_density_bin, cmap="seismic")
        plt.title("Target binary")
        plt.tight_layout()
        plt.close(fig)

        for (a, b, c), loss_name in zip(LOSS_COMB, LOSS_NAMES):
            for subject in subjects:
                data_dir = DATA_ROOT
                if not data_dir.exists():
                    print(f"[skip] subject {subject}: data dir not found -> {data_dir}")
                    continue

                try:
                    polar_map = nib.load(str(data_dir / FNAME_ANG)).get_fdata()
                    ecc_map = nib.load(str(data_dir / FNAME_ECC)).get_fdata()
                    sigma_map = nib.load(str(data_dir / FNAME_SIGMA)).get_fdata()
                    aparc_roi = nib.load(str(data_dir / FNAME_APARC)).get_fdata()
                    label_map = nib.load(str(data_dir / FNAME_LABEL)).get_fdata()
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

                median_lh = [np.median(cs_coords_lh[0]), np.median(cs_coords_lh[1]), np.median(cs_coords_lh[2])]
                median_rh = [np.median(cs_coords_rh[0]), np.median(cs_coords_rh[1]), np.median(cs_coords_rh[2])]

                print(f"target: {target_name}")
                print(f"loss: {loss_name}")
                print(f"a,b,c: {a},{b},{c}")

                hemi_iter: Iterable[tuple[np.ndarray, str, list[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Integer]] = [
                    (gm_lh, "LH", median_lh, good_coords_lh, v1_coords_lh, v2_coords_lh, v3_coords_lh, DIM_BETA_LH),
                    (gm_rh, "RH", median_rh, good_coords_rh, v1_coords_rh, v2_coords_rh, v3_coords_rh, DIM_BETA_RH),
                ]

                for gm_mask, hem, start_location, good_h, v1_h, v2_h, v3_h, dim_beta in hemi_iter:
                    data_id = f"{subject}_{hem}_V1_n1000_1x10_{loss_name}_{THRESH}_{target_name}"
                    pkl_path = output_root / f"{data_id}.pkl"
                    txt_path = output_root / f"{data_id}.txt"
                    png_path = output_root / f"{data_id}_phosphene_map.png"

                    if pkl_path.exists():
                        print(f"{subject} {hem} {target_name} {loss_name} already processed.")
                        continue

                    dimensions = [DIM_ALPHA, dim_beta, DIM_OFFSET, DIM_SHANK]
                    lhs2 = cook_initial_point_generator("lhs", criterion="maximin")

                    @use_named_args(dimensions=dimensions)
                    def objective(alpha, beta, offset_from_base, shank_length):
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

                        phos_v1 = get_phosphenes(contacts_xyz_moved, v1_h, polar_map, ecc_map, sigma_map)
                        if phos_v1.size == 0:
                            print("    3.00 0.00 0.00 1.00 False")
                            return 3.0

                        m_inv = 1 / get_cortical_magnification(phos_v1[:, 1], CORT_MAG_MODEL)
                        spread = cortical_spread(AMP)
                        sigmas = (spread * m_inv) / 2
                        phos_v1[:, 2] = sigmas

                        phosphene_map = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype="float32")
                        phosphene_map = prf_to_phos(phosphene_map, phos_v1, view_angle=VIEW_ANGLE, phSizeScale=1)

                        max_ph = np.max(phosphene_map)
                        sum_ph = np.sum(phosphene_map)
                        if max_ph > 0:
                            phosphene_map /= max_ph
                        if sum_ph > 0:
                            phosphene_map /= np.sum(phosphene_map)

                        dice, _, _ = DC(target_density, phosphene_map, np.percentile(target_density, DC_PERCENTILE))
                        grid_yield = get_yield(contacts_xyz_moved, good_h)
                        hell_d = hellinger_distance(phosphene_map.flatten(), target_density.flatten())

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

                        print(
                            "    ",
                            f"{cost:.2f}",
                            f"{dice:.2f}",
                            f"{grid_yield:.2f}",
                            f"{par3:.2f}",
                            grid_valid,
                        )
                        return float(cost)

                    res = gp_minimize(
                        objective,
                        x0=X0,
                        dimensions=dimensions,
                        n_jobs=-1,
                        n_calls=NUM_CALLS,
                        n_initial_points=NUM_INITIAL_POINTS,
                        initial_point_generator=lhs2,
                        callback=[custom_stopper],
                    )

                    print(
                        f"subject {subject} {hem}, best alpha: {res.x[0]}, "
                        f"best beta: {res.x[1]}, best offset_from_base: {res.x[2]}, best shank_length: {res.x[3]}"
                    )

                    # Recompute map/metrics for best parameters
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

                    phos_all = get_phosphenes(contacts_xyz_moved, good_h, polar_map, ecc_map, sigma_map)
                    phos_v1 = get_phosphenes(contacts_xyz_moved, v1_h, polar_map, ecc_map, sigma_map)
                    phos_v2 = get_phosphenes(contacts_xyz_moved, v2_h, polar_map, ecc_map, sigma_map)
                    phos_v3 = get_phosphenes(contacts_xyz_moved, v3_h, polar_map, ecc_map, sigma_map)

                    if phos_v1.size > 0:
                        m_inv = 1 / get_cortical_magnification(phos_v1[:, 1], CORT_MAG_MODEL)
                        spread = cortical_spread(AMP)
                        phos_v1[:, 2] = (spread * m_inv) / 2

                    phosphene_map = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype="float32")
                    phosphene_map = prf_to_phos(phosphene_map, phos_v1, view_angle=VIEW_ANGLE, phSizeScale=1)
                    if np.max(phosphene_map) > 0:
                        phosphene_map /= np.max(phosphene_map)
                    if np.sum(phosphene_map) > 0:
                        phosphene_map /= np.sum(phosphene_map)

                    dice, _, _ = DC(target_density, phosphene_map, np.percentile(target_density, DC_PERCENTILE))
                    grid_yield = get_yield(contacts_xyz_moved, good_h)
                    hell_d = hellinger_distance(phosphene_map.flatten(), target_density.flatten())
                    print(f"best dice, yield, KL: {dice}, {grid_yield}, {hell_d}")

                    map_bin = phosphene_map > np.percentile(phosphene_map, DC_PERCENTILE)
                    fig, _ = plt.subplots(1, 2, figsize=(10, 4))
                    plt.subplot(1, 2, 1).imshow(map_bin, cmap="seismic", vmin=0, vmax=max(np.max(phosphene_map) / 100, 1e-6))
                    plt.title("Binarized map")
                    plt.subplot(1, 2, 2).imshow(phosphene_map, cmap="seismic", vmin=0, vmax=max(np.max(phosphene_map) / 100, 1e-6))
                    plt.title("Raw phosphene map")
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    with pkl_path.open("wb") as f:
                        pickle.dump(
                            [
                                res,
                                grid_valid,
                                dice,
                                hell_d,
                                grid_yield,
                                contacts_xyz_moved,
                                good_h,
                                v1_h,
                                v2_h,
                                v3_h,
                                phos_all,
                                phos_v1,
                                phos_v2,
                                phos_v3,
                            ],
                            f,
                            protocol=-1,
                        )
                    save_result_text(
                        txt_path=txt_path,
                        subject=subject,
                        hemisphere=hem,
                        target_name=target_name,
                        loss_name=loss_name,
                        best_x=res.x,
                        grid_valid=grid_valid,
                        dice=dice,
                        hell_d=hell_d,
                        grid_yield=grid_yield,
                    )
                    print(f"saved: {pkl_path}")


if __name__ == "__main__":
    run()
