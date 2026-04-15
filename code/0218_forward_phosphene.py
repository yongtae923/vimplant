# code/0218_forward_phosphene.py
"""
Forward pass only: given 4 implant parameters and a directory with retinotopy files,
compute and save the phosphene map. No Bayesian optimization, no loss.
Uses the same pipeline as 0216_opt (grid, implant, PRF->phosphene) without skopt.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress known warnings (same as 0216_opt)
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

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASECODE_DIR = PROJECT_ROOT / "basecode"
if BASECODE_DIR.exists():
    sys.path.insert(0, str(BASECODE_DIR))
else:
    raise FileNotFoundError(f"basecode directory not found: {BASECODE_DIR}")

from ninimplant import get_xyz
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

# Retinotopy file names (5 mgz files; same as 0216_opt)
FNAME_ANG = "inferred_angle.mgz"
FNAME_ECC = "inferred_eccen.mgz"
FNAME_SIGMA = "inferred_sigma.mgz"
FNAME_APARC = "aparc+aseg.mgz"
FNAME_LABEL = "inferred_varea.mgz"

# Pipeline constants (match 0216_opt)
WINDOWSIZE = 1000
N_CONTACTPOINTS_SHANK = 10
SPACING_ALONG_XY = 1
CORT_MAG_MODEL = "wedge-dipole"
VIEW_ANGLE = 90
AMP = 100


def coords_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((3, 0), dtype=np.int32)
    aset = set(map(tuple, np.round(a).T.astype(np.int32)))
    bset = set(map(tuple, np.round(b).T.astype(np.int32)))
    inter = list(aset & bset)
    if not inter:
        return np.empty((3, 0), dtype=np.int32)
    return np.array(inter, dtype=np.int32).T


def normalize_phosphene_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64, copy=True)
    max_val = np.max(arr)
    if max_val > 0:
        arr /= max_val
    total = np.sum(arr)
    if total > 0:
        arr /= total
    return arr.astype(np.float32)


def load_retinotopy(data_dir: Path):
    """Load 5 retinotopy/atlas files from data_dir. Returns maps and derived coords for LH/RH."""
    data_dir = Path(data_dir)
    polar_map = nib.load(str(data_dir / FNAME_ANG)).get_fdata()
    ecc_map = nib.load(str(data_dir / FNAME_ECC)).get_fdata()
    sigma_map = nib.load(str(data_dir / FNAME_SIGMA)).get_fdata()
    aparc_roi = nib.load(str(data_dir / FNAME_APARC)).get_fdata()
    label_map = nib.load(str(data_dir / FNAME_LABEL)).get_fdata()

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
    good_coords_lh = coords_intersection(good_coords, np.asarray(gm_coords_lh))
    good_coords_rh = coords_intersection(good_coords, np.asarray(gm_coords_rh))
    v1_coords_lh = coords_intersection(v1_coords, np.asarray(gm_coords_lh))
    v1_coords_rh = coords_intersection(v1_coords, np.asarray(gm_coords_rh))

    median_lh = [np.median(cs_coords_lh[0]), np.median(cs_coords_lh[1]), np.median(cs_coords_lh[2])]
    median_rh = [np.median(cs_coords_rh[0]), np.median(cs_coords_rh[1]), np.median(cs_coords_rh[2])]

    return {
        "polar_map": polar_map,
        "ecc_map": ecc_map,
        "sigma_map": sigma_map,
        "gm_lh": gm_lh,
        "gm_rh": gm_rh,
        "median_lh": median_lh,
        "median_rh": median_rh,
        "good_coords_lh": good_coords_lh,
        "good_coords_rh": good_coords_rh,
        "v1_coords_lh": v1_coords_lh,
        "v1_coords_rh": v1_coords_rh,
    }


def make_phosphene_map(
    data_dir: Path,
    alpha: float,
    beta: float,
    offset_from_base: float,
    shank_length: float,
    hemisphere: str = "LH",
) -> np.ndarray:
    """
    Single forward pass: 4 parameters + retinotopy data -> phosphene map (WINDOWSIZE x WINDOWSIZE).
    hemisphere: "LH" or "RH"
    """
    data = load_retinotopy(data_dir)
    if hemisphere.upper() == "LH":
        gm_mask = data["gm_lh"]
        start_location = data["median_lh"]
        v1_h = data["v1_coords_lh"]
    else:
        gm_mask = data["gm_rh"]
        start_location = data["median_rh"]
        v1_h = data["v1_coords_rh"]

    polar_map = data["polar_map"]
    ecc_map = data["ecc_map"]
    sigma_map = data["sigma_map"]

    new_angle = (float(alpha), float(beta), 0.0)
    orig_grid = create_grid(
        start_location,
        shank_length=float(shank_length),
        n_contactpoints_shank=N_CONTACTPOINTS_SHANK,
        spacing_along_xy=SPACING_ALONG_XY,
        offset_from_origin=0,
    )
    _, contacts_xyz_moved, _, _, _, _, _, _, _ = implant_grid(
        gm_mask, orig_grid, start_location, new_angle, float(offset_from_base)
    )

    phos_v1 = get_phosphenes(contacts_xyz_moved, v1_h, polar_map, ecc_map, sigma_map)
    if phos_v1.size == 0:
        return np.zeros((WINDOWSIZE, WINDOWSIZE), dtype=np.float32)

    m_inv = 1 / get_cortical_magnification(phos_v1[:, 1], CORT_MAG_MODEL)
    spread = cortical_spread(AMP)
    phos_v1[:, 2] = (spread * m_inv) / 2

    phosphene_map = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype=np.float32)
    phosphene_map = prf_to_phos(phosphene_map, phos_v1, view_angle=VIEW_ANGLE, phSizeScale=1)
    return normalize_phosphene_map(phosphene_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward pass: 4 params + retinotopy dir -> phosphene map only (no loss)."
    )
    parser.add_argument(
        "alpha",
        type=float,
        help="Implant angle alpha (e.g. -90 to 90)",
    )
    parser.add_argument(
        "beta",
        type=float,
        help="Implant angle beta (LH: -15~110, RH: -110~15)",
    )
    parser.add_argument(
        "offset_from_base",
        type=float,
        help="Offset from base (e.g. 0 to 40)",
    )
    parser.add_argument(
        "shank_length",
        type=float,
        help="Shank length (e.g. 10 to 40)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "input" / "100610",
        help="Directory containing retinotopy mgz files (default: data/input/100610)",
    )
    parser.add_argument(
        "--hemisphere",
        choices=["LH", "RH"],
        default="LH",
        help="Hemisphere (default: LH)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for phosphene map (.npy). Default: data/output/forward_phosphene/<hemi>_phosphene.npy",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save a PNG visualization",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data dir not found: {data_dir}")
        sys.exit(1)
    for f in [FNAME_ANG, FNAME_ECC, FNAME_SIGMA, FNAME_APARC, FNAME_LABEL]:
        if not (data_dir / f).exists():
            print(f"Error: missing file {f} in {data_dir}")
            sys.exit(1)

    phosphene_map = make_phosphene_map(
        data_dir=data_dir,
        alpha=args.alpha,
        beta=args.beta,
        offset_from_base=args.offset_from_base,
        shank_length=args.shank_length,
        hemisphere=args.hemisphere,
    )

    out_path = args.output
    if out_path is None:
        out_dir = PROJECT_ROOT / "data" / "output" / "forward_phosphene"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.hemisphere}_alpha{args.alpha}_beta{args.beta}_off{args.offset_from_base}_shank{args.shank_length}_phosphene.npy"
    else:
        out_path = Path(out_path)
        if out_path.suffix.lower() != ".npy":
            out_path = out_path.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_path, phosphene_map)
    print(f"Saved phosphene map: {out_path}")

    if args.save_png:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(phosphene_map, cmap="seismic")
        ax.set_title(f"Phosphene map ({args.hemisphere})")
        ax.axis("off")
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
