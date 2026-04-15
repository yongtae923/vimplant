# make_phosphene_target.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT_NPZ  = Path(r"D:\yongtae\vimplant\data\letters\mnist10.npz")
OUTPUT_DIR = Path(r"D:\yongtae\vimplant\data\letters\phosphene_targets")

SAMPLE_INDEX = 0  # 0~9


def normalize_density(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    max_v = float(out.max())
    if max_v > 0:
        out /= max_v
    sum_v = float(out.sum())
    if sum_v > 0:
        out /= sum_v
    return out


def main() -> None:
    with np.load(INPUT_NPZ, allow_pickle=False) as data:
        phos = data["test_phosphenes"]

    if phos.ndim == 4 and phos.shape[1] == 1:
        phos = phos[:, 0, :, :]  # (10, H, W)

    img = phos[SAMPLE_INDEX].astype(np.float32)
    target = normalize_density(img)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = OUTPUT_DIR / f"phosphene_target_{SAMPLE_INDEX:03d}.npy"
    png_path = OUTPUT_DIR / f"phosphene_target_{SAMPLE_INDEX:03d}.png"

    np.save(npy_path, target)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(target, cmap="seismic")
    ax.set_title(f"Phosphene target (index={SAMPLE_INDEX})", fontsize=12)
    ax.axis("off")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {npy_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()