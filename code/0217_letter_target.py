from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


WINDOWSIZE = 1000
LETTERS_TO_SAVE = ["A", "B", "C"]
ABC_LETTERS = ["A", "B", "C"]
AZ_LETTERS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
LETTER_HEIGHT_RATIO = 0.56
THIN_ITERATIONS = 14
LINE_GAUSSIAN_FWHM = 40

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "letter_target"


def gaussian_from_line_mask(mask: np.ndarray, fwhm: float) -> np.ndarray:
    """
    Build a gaussian field using letter-line mask as source (not image center).
    """
    sigma = max(float(fwhm) / 2.354820045, 1e-6)
    h, w = mask.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    fxx, fyy = np.meshgrid(fx, fy)
    transfer = np.exp(-(2.0 * (np.pi**2) * (sigma**2) * (fxx**2 + fyy**2))).astype(np.float32)

    field = np.fft.ifft2(np.fft.fft2(mask.astype(np.float32)) * transfer).real.astype(np.float32)
    field[field < 0] = 0.0
    return field


def normalize_density(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    max_v = float(out.max()) if out.size > 0 else 0.0
    if max_v > 0:
        out /= max_v
    sum_v = float(out.sum())
    if sum_v > 0:
        out /= sum_v
    return out


def _binary_erosion(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = eroded
    return out


def _centered_letter_path(letter: str, windowsize: int, height_ratio: float) -> MplPath:
    font = FontProperties(family="DejaVu Sans", weight="bold")
    base = TextPath((0, 0), letter, size=1.0, prop=font)
    bbox = base.get_extents()
    target_h = windowsize * height_ratio
    scale = target_h / max(float(bbox.height), 1e-6)

    scaled_w = float(bbox.width) * scale
    scaled_h = float(bbox.height) * scale
    tx = (windowsize - scaled_w) * 0.5 - float(bbox.x0) * scale
    ty = (windowsize - scaled_h) * 0.5 - float(bbox.y0) * scale

    trans = Affine2D().scale(scale).translate(tx, ty)
    return trans.transform_path(base)


def _rasterize_path_evenodd(path: MplPath, windowsize: int) -> np.ndarray:
    yy, xx = np.mgrid[0:windowsize, 0:windowsize]
    points = np.column_stack(
        [
            xx.ravel().astype(np.float32) + 0.5,
            (windowsize - 1 - yy.ravel()).astype(np.float32) + 0.5,
        ]
    )

    # Use even-odd fill over sub-polygons to preserve glyph holes (e.g., A/B).
    mask = np.zeros(points.shape[0], dtype=bool)
    for poly in path.to_polygons():
        if len(poly) < 3:
            continue
        poly_path = MplPath(poly)
        mask ^= poly_path.contains_points(points)
    return mask.reshape(windowsize, windowsize)


def make_letter_target(letter: str) -> np.ndarray:
    letter_path = _centered_letter_path(letter, WINDOWSIZE, LETTER_HEIGHT_RATIO)
    mask = _rasterize_path_evenodd(letter_path, WINDOWSIZE)

    thin_mask = _binary_erosion(mask, THIN_ITERATIONS)

    # Apply gaussian to letter line first.
    line_gauss = gaussian_from_line_mask(thin_mask, LINE_GAUSSIAN_FWHM)

    # Then apply left-half masking.
    center_x = WINDOWSIZE // 2
    line_gauss[:, :center_x] = 0.0

    target = line_gauss
    return normalize_density(target)


def save_target_image(target: np.ndarray, letter: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"targ-letter-{letter}.npy"
    png_path = out_dir / f"targ-letter-{letter}.png"

    np.save(npy_path, target.astype(np.float32))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(target, cmap="seismic")
    ax.set_title(f"Letter target: {letter}", fontsize=12)
    ax.axis("off")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_target(letters: list[str]) -> np.ndarray:
    combined = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype=np.float32)
    for letter in letters:
        combined += make_letter_target(letter).astype(np.float32)
    return normalize_density(combined)


def main() -> None:
    print(f"output dir: {OUTPUT_DIR}")
    for letter in LETTERS_TO_SAVE:
        target = make_letter_target(letter)
        save_target_image(target, letter, OUTPUT_DIR)
        print(f"[ok] saved letter target: {letter}")

    combined_target_abc = make_combined_target(ABC_LETTERS)
    save_target_image(combined_target_abc, "ABC", OUTPUT_DIR)
    print("[ok] saved letter target: ABC")

    combined_target_az = make_combined_target(AZ_LETTERS)
    save_target_image(combined_target_az, "AZ", OUTPUT_DIR)
    print("[ok] saved letter target: AZ")


if __name__ == "__main__":
    main()
