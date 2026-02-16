from __future__ import annotations

import math
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT_MULTI = Path("data/1110_multi_electrode_100610")
DATA_ROOT_BASE = Path("data/output/100610")
MULTI_GRID_KEY = "5x10x10+5x10x10"
BASE_GRID_TAG = "1x10"
OUTPUT_DIR = Path("data/output/100610_multi_vs_single")


class CompatUnpickler(pickle.Unpickler):
    """PKL 호환성을 위해 누락된 의존성을 최소한으로 채우는 Unpickler."""

    _STUBS = {
        "f": staticmethod(lambda *args, **kwargs: None),
        "custom_stopper": staticmethod(lambda *args, **kwargs: False),
    }

    def find_class(self, module: str, name: str):
        if module == "__main__" and name in self._STUBS:
            return self._STUBS[name]
        return super().find_class(module, name)


def ensure_numpy_aliases() -> None:
    core_mod_name = "numpy._core"
    if core_mod_name not in sys.modules:
        core_pkg = types.ModuleType(core_mod_name)
        core_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[core_mod_name] = core_pkg
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.overrides", np.core.overrides)


def safe_pickle_load(path: Path):
    ensure_numpy_aliases()
    with path.open("rb") as f:
        return CompatUnpickler(f).load()


@dataclass(frozen=True)
class EntryMeta:
    subject: str
    hemisphere: str
    target: str
    grid_tag: str
    path: Path


@dataclass
class ResultRecord:
    meta: EntryMeta
    fun: float
    params_a: List[float]
    params_b: Optional[List[float]]
    metrics: Dict[str, float]
    x_dim: int


PARAM_LABELS = ["Alpha", "Beta", "Offset", "Shank length"]
METRIC_LABELS = ["Loss", "Dice coefficient", "Hellinger distance", "Grid yield"]
_TXT_CACHE: Dict[str, Optional[Path]] = {}


def parse_meta(path: Path) -> EntryMeta:
    stem_parts = path.stem.split("_")
    if len(stem_parts) < 8:
        raise ValueError(f"예상치 못한 파일명 형식: {path.name}")
    subject, hemisphere = stem_parts[0], stem_parts[1]
    grid_tag = stem_parts[4]
    target = stem_parts[-1].split("-")[-1]
    return EntryMeta(subject=subject, hemisphere=hemisphere, target=target, grid_tag=grid_tag, path=path)


def find_metrics_txt(path: Path) -> Optional[Path]:
    base_name = path.stem + ".txt"
    if base_name in _TXT_CACHE:
        return _TXT_CACHE[base_name]

    candidates = [
        path.with_suffix(".txt"),
        DATA_ROOT_MULTI / base_name,
        DATA_ROOT_BASE / base_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            _TXT_CACHE[base_name] = candidate
            return candidate

    data_root = Path("data")
    try:
        for candidate in data_root.rglob(base_name):
            if candidate.exists():
                _TXT_CACHE[base_name] = candidate
                return candidate
    except Exception:
        pass

    _TXT_CACHE[base_name] = None
    return None


def load_metrics(path: Path, default_loss: float) -> Dict[str, float]:
    metrics = {
        "loss": float(default_loss),
        "dice": np.nan,
        "hell": np.nan,
        "yield": np.nan,
    }
    txt_path = find_metrics_txt(path)
    if not txt_path:
        return metrics

    try:
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("dice coefficient"):
                    metrics["dice"] = float(line.split(":")[1].strip())
                elif line.lower().startswith("hellinger distance"):
                    metrics["hell"] = float(line.split(":")[1].strip())
                elif line.lower().startswith("grid yield"):
                    metrics["yield"] = float(line.split(":")[1].strip())
    except Exception:
        pass
    return metrics


def extract_best_record(path: Path) -> ResultRecord:
    data = safe_pickle_load(path)
    if not isinstance(data, Iterable):
        raise TypeError(f"{path} : 리스트 형태의 skopt 결과를 기대했지만 {type(data)} 입니다.")

    best_entry: Optional[Dict] = None
    best_fun = math.inf
    for item in data:
        if isinstance(item, dict) and "fun" in item and "x" in item:
            fun = item["fun"]
            x = item["x"]
            if fun is None or x is None:
                continue
            try:
                fun_val = float(fun)
            except Exception:
                continue
            if fun_val < best_fun:
                best_fun = fun_val
                best_entry = item

    if best_entry is None:
        raise ValueError(f"{path} 에서 유효한 최적화 결과를 찾지 못했습니다.")

    x_vals = list(map(float, best_entry["x"]))
    if len(x_vals) >= 8:
        params_a = [x_vals[0], x_vals[1], x_vals[2], x_vals[3]]
        params_b = [x_vals[4], x_vals[5], x_vals[6], x_vals[7]]
    elif len(x_vals) >= 4:
        params_a = [x_vals[0], x_vals[1], x_vals[2], x_vals[3]]
        params_b = None
    else:
        raise ValueError(f"{path} 의 x 길이({len(x_vals)})가 4 미만입니다.")

    meta = parse_meta(path)
    metrics = load_metrics(path, best_fun)

    return ResultRecord(meta=meta, fun=best_fun, params_a=params_a, params_b=params_b, metrics=metrics, x_dim=len(x_vals))


def collect_multi_entries() -> List[EntryMeta]:
    entries: List[EntryMeta] = []
    for pkl_path in sorted(DATA_ROOT_MULTI.glob("*.pkl")):
        meta = parse_meta(pkl_path)
        if MULTI_GRID_KEY not in meta.grid_tag:
            continue
        entries.append(meta)
    return entries


def matching_base_path(meta: EntryMeta) -> Optional[Path]:
    candidate = DATA_ROOT_BASE / f"{meta.subject}_{meta.hemisphere}_V1_n1000_{BASE_GRID_TAG}_dice-yield-HD_0.05_targ-{meta.target}.pkl"
    return candidate if candidate.exists() else None


def build_comparisons() -> Dict[Tuple[str, str], List[Tuple[str, ResultRecord, ResultRecord]]]:
    comparisons: Dict[Tuple[str, str], List[Tuple[str, ResultRecord, ResultRecord]]] = {}
    for meta in collect_multi_entries():
        base_path = matching_base_path(meta)
        if base_path is None:
            print(f"[경고] 기준 파일을 찾지 못했습니다: {meta.subject} {meta.hemisphere} {meta.target}")
            continue
        multi_record = extract_best_record(meta.path)
        base_record = extract_best_record(base_path)
        key = (meta.subject, meta.hemisphere)
        comparisons.setdefault(key, []).append((meta.target, multi_record, base_record))
    return comparisons


def _plot_grouped_bars(
    ax: plt.Axes,
    labels: List[str],
    series_values: List[List[float]],
    series_labels: List[str],
    colors: List[str],
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.35
    used_labels: set[str] = set()
    for idx, (values, label, color) in enumerate(zip(series_values, series_labels, colors)):
        offsets = x + (idx - (len(series_values) - 1) / 2) * width
        for ox, val in zip(offsets, values):
            if np.isnan(val):
                continue
            lbl = None if label in used_labels else label
            bar = ax.bar(ox, val, width=width, label=lbl, color=color)
            used_labels.add(label)
            ax.bar_label(bar, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)


def plot_comparisons(comparisons: Dict[Tuple[str, str], List[Tuple[str, ResultRecord, ResultRecord]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    color_multi = "#1f77b4"
    color_base = "#ff7f0e"

    for (subject, hemisphere), items in comparisons.items():
        items_sorted = sorted(items, key=lambda t: t[0])
        n_targets = len(items_sorted)
        if n_targets == 0:
            continue
        fig, axes = plt.subplots(n_targets, 3, figsize=(18, 3.4 * n_targets), squeeze=False)
        fig.suptitle(f"{subject} {hemisphere} - 5x10x10 vs {BASE_GRID_TAG} comparison", fontsize=16, fontweight="bold")

        for row, (target, multi_record, base_record) in enumerate(items_sorted):
            metric_ax = axes[row, 0]
            probe_a_ax = axes[row, 1]
            probe_b_ax = axes[row, 2]

            # Loss & related metrics subplot
            metrics_multi = [
                multi_record.metrics["loss"],
                multi_record.metrics["dice"],
                multi_record.metrics["hell"],
                multi_record.metrics["yield"],
            ]
            metrics_base = [
                base_record.metrics["loss"],
                base_record.metrics["dice"],
                base_record.metrics["hell"],
                base_record.metrics["yield"],
            ]
            _plot_grouped_bars(
                metric_ax,
                METRIC_LABELS,
                [metrics_multi, metrics_base],
                ["5x10x10+5x10x10", BASE_GRID_TAG],
                [color_multi, color_base],
            )
            metric_ax.set_title(f"{target} - Loss & metrics", fontsize=12, fontweight="bold")

            # Probe A parameters
            _plot_grouped_bars(
                probe_a_ax,
                PARAM_LABELS,
                [multi_record.params_a, base_record.params_a],
                ["5x10x10+5x10x10", BASE_GRID_TAG],
                [color_multi, color_base],
            )
            probe_a_ax.set_title(f"{target} - Probe A parameters", fontsize=12, fontweight="bold")

            # Probe B parameters (base may be missing)
            params_base_b = base_record.params_b if base_record.params_b is not None else [np.nan] * len(PARAM_LABELS)
            params_multi_b = multi_record.params_b if multi_record.params_b is not None else [np.nan] * len(PARAM_LABELS)
            _plot_grouped_bars(
                probe_b_ax,
                PARAM_LABELS,
                [params_multi_b, params_base_b],
                ["5x10x10+5x10x10", BASE_GRID_TAG],
                [color_multi, color_base],
            )
            probe_b_ax.set_title(f"{target} - Probe B parameters", fontsize=12, fontweight="bold")

        handles, labels = axes[0, 1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = OUTPUT_DIR / f"{subject}_{hemisphere}_comparison.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"[saved] {output_path}")


def main():
    comparisons = build_comparisons()
    if not comparisons:
        print("비교할 대상이 없습니다. 디렉터리 경로와 파일명을 다시 확인하세요.")
        return
    plot_comparisons(comparisons)


if __name__ == "__main__":
    main()


