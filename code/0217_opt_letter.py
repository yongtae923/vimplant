from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
BASE_SCRIPT_PATH = CODE_DIR / "0216_opt.py"
PLOT_SCRIPT_PATH = CODE_DIR / "0217_opt_plot.py"
SELECT_SCRIPT_PATH = CODE_DIR / "0217_opt_select.py"
LETTER_TARGET_DIR = PROJECT_ROOT / "data" / "output" / "letter_target"
OUTPUT_ROOT_LETTER = PROJECT_ROOT / "data" / "output" / "opt_npz_letter"
OUTPUT_PLOT_LETTER = PROJECT_ROOT / "data" / "output" / "opt_plot_letter"
OUTPUT_SELECT_NPZ_LETTER = PROJECT_ROOT / "data" / "output" / "opt_select_npz_letter"
OUTPUT_SELECT_PLOT_LETTER = PROJECT_ROOT / "data" / "output" / "opt_select_plot_letter"

# Pipeline switches
RUN_OPTIMIZATION = True
RUN_PLOT = True
RUN_SELECT = True

# Only these 5 targets are used for letter optimization.
LETTER_TARGET_FILES: list[tuple[str, str]] = [
    ("targ-letter-A", "targ-letter-A.npy"),
    ("targ-letter-B", "targ-letter-B.npy"),
    ("targ-letter-C", "targ-letter-C.npy"),
    ("targ-letter-ABC", "targ-letter-ABC.npy"),
    ("targ-letter-AZ", "targ-letter-AZ.npy"),
]


def _load_module(script_path: Path, module_name: str) -> ModuleType:
    if not script_path.exists():
        raise FileNotFoundError(f"script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create import spec for: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_letter_targets(base_module: ModuleType) -> tuple[list[np.ndarray], list[str]]:
    maps: list[np.ndarray] = []
    names: list[str] = []

    for target_name, fname in LETTER_TARGET_FILES:
        fpath = LETTER_TARGET_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"missing letter target file: {fpath}")

        arr = np.load(fpath)
        if arr.ndim != 2:
            raise ValueError(f"target map must be 2D: {fpath} (shape={arr.shape})")

        if arr.shape != (int(base_module.WINDOWSIZE), int(base_module.WINDOWSIZE)):
            raise ValueError(
                f"target map shape mismatch: {fpath} "
                f"(shape={arr.shape}, expected={(base_module.WINDOWSIZE, base_module.WINDOWSIZE)})"
            )

        arr = base_module.normalize_density(np.asarray(arr, dtype=np.float64))
        maps.append(arr)
        names.append(target_name)

    return maps, names


def run_optimization() -> None:
    base_module = _load_module(BASE_SCRIPT_PATH, "opt_base_0216")
    target_maps, target_names = _load_letter_targets(base_module)

    # Keep optimization logic identical to 0216_opt.py and only swap target set/output path.
    base_module.TARGET_MAPS = target_maps
    base_module.TARGET_NAMES = target_names
    base_module.OUTPUT_ROOT = OUTPUT_ROOT_LETTER

    print(f"letter target dir: {LETTER_TARGET_DIR}")
    print(f"letter output npz dir: {OUTPUT_ROOT_LETTER}")
    print(f"letter targets: {', '.join(target_names)}")
    base_module.run()


def run_plot() -> None:
    plot_module = _load_module(PLOT_SCRIPT_PATH, "opt_plot_0217")
    plot_module.NPZ_DIR = OUTPUT_ROOT_LETTER
    plot_module.OUTPUT_DIR = OUTPUT_PLOT_LETTER
    print(f"letter plot input npz dir: {plot_module.NPZ_DIR}")
    print(f"letter plot output dir: {plot_module.OUTPUT_DIR}")
    plot_module.main()


def run_select() -> None:
    select_module = _load_module(SELECT_SCRIPT_PATH, "opt_select_0217")
    select_module.INPUT_NPZ_DIR = OUTPUT_ROOT_LETTER
    select_module.OUTPUT_NPZ_DIR = OUTPUT_SELECT_NPZ_LETTER
    select_module.OUTPUT_PLOT_DIR = OUTPUT_SELECT_PLOT_LETTER
    print(f"letter select input npz dir: {select_module.INPUT_NPZ_DIR}")
    print(f"letter select output npz dir: {select_module.OUTPUT_NPZ_DIR}")
    print(f"letter select output plot dir: {select_module.OUTPUT_PLOT_DIR}")
    select_module.main()


def run() -> None:
    print("=== letter optimization pipeline ===")
    if RUN_OPTIMIZATION:
        run_optimization()
    if RUN_PLOT:
        run_plot()
    if RUN_SELECT:
        run_select()
    print("=== done ===")


if __name__ == "__main__":
    run()
