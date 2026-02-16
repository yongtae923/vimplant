# -*- coding: utf-8 -*-
"""
Compare outputs for subject 102311 between single-probe (py) and dual-probe (double).

Reads metrics from corresponding .txt files and phosphene map images, builds
side-by-side figures and metric bar charts, and saves comparisons into the
double output folder for convenient review.
"""

import os
import argparse
import re
import glob
import json
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread


# Root paths (adjust if needed)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_PY = os.path.join(PROJECT_ROOT, 'data', 'output', '102311_py')
OUT_DOUBLE = os.path.join(PROJECT_ROOT, 'data', 'output', '102311_double')

# Loss weights (must match experiment)
LOSS_A = 1.0
LOSS_B = 0.1
LOSS_C = 1.0


@dataclass
class TrialKey:
    hemisphere: str   # 'LH' or 'RH'
    target: str       # e.g., 'targ-upper', 'targ-full', etc.


@dataclass
class TrialMetrics:
    dice: float
    hell: float
    yield_: float
    params: Dict[str, str]
    txt_path: str
    img_path: Optional[str]


_TXT_PATTERN = re.compile(r"^(?P<subj>\d+)_(?P<hemi>LH|RH)_V1_n1000_1x10_.+?_(?P<thresh>[^_]+)_(?P<target>targ-[^\.]+)\.txt$")
_IMG_SUFFIX = '_phosphene_map.png'


def _parse_metrics_txt(txt_path: str) -> Optional[TrialMetrics]:
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            # keep leading spaces; only strip per-line when matching keys
            lines = [ln.rstrip('\r\n') for ln in f.readlines()]
        vals: Dict[str, str] = {}
        def _to_float(val: str) -> Optional[float]:
            try:
                return float(val.strip())
            except Exception:
                return None

        for ln in lines:
            s = ln.strip()
            if s.startswith('Dice coefficient:'):
                f = _to_float(s.split(':', 1)[1])
                if f is not None:
                    vals['dice'] = f
            elif s.startswith('Hellinger distance:'):
                f = _to_float(s.split(':', 1)[1])
                if f is not None:
                    vals['hell'] = f
            elif s.startswith('Grid yield:'):
                f = _to_float(s.split(':', 1)[1])
                if f is not None:
                    vals['yield'] = f

        params: Dict[str, str] = {}
        for ln in lines:
            s = ln.strip()
            if 'Best parameters:' in s:
                continue
            if s.startswith('Alpha:') or s.startswith('Beta:') or \
               s.startswith('Offset') or s.startswith('Shank length:') or \
               s.startswith('Probe '):
                parts = s.split(':', 1)
                if len(parts) == 2:
                    params[parts[0].strip()] = parts[1].strip()

        tm = TrialMetrics(
            dice=float(vals.get('dice', np.nan)),
            hell=float(vals.get('hell', np.nan)),
            yield_=float(vals.get('yield', np.nan)),
            params=params,
            txt_path=txt_path,
            img_path=None,
        )
        print(f"[DEBUG] Parsed: {txt_path}")
        print(f"        dice={tm.dice}, yield={tm.yield_}, hell={tm.hell}")
        return tm
    except Exception:
        return None


def _build_key_from_filename(path: str) -> Optional[TrialKey]:
    base = os.path.basename(path)
    m = _TXT_PATTERN.match(base)
    if not m:
        return None
    return TrialKey(hemisphere=m.group('hemi'), target=m.group('target'))


def _pair_trials() -> List[Tuple[TrialKey, Optional[TrialMetrics], Optional[TrialMetrics]]]:
    txt_py = glob.glob(os.path.join(OUT_PY, '*.txt'))
    txt_double = glob.glob(os.path.join(OUT_DOUBLE, '*.txt'))

    py_map: Dict[Tuple[str, str], TrialMetrics] = {}
    for p in txt_py:
        k = _build_key_from_filename(p)
        if not k:
            continue
        tm = _parse_metrics_txt(p)
        if tm:
            py_map[(k.hemisphere, k.target)] = tm

    double_map: Dict[Tuple[str, str], TrialMetrics] = {}
    for p in txt_double:
        k = _build_key_from_filename(p)
        if not k:
            continue
        tm = _parse_metrics_txt(p)
        if tm:
            double_map[(k.hemisphere, k.target)] = tm

    # Attach image paths
    def _img_for_txt(txt_path: str) -> Optional[str]:
        base = os.path.splitext(txt_path)[0]
        img_path = base + _IMG_SUFFIX
        return img_path if os.path.exists(img_path) else None

    for m in py_map.values():
        m.img_path = _img_for_txt(m.txt_path)
    for m in double_map.values():
        m.img_path = _img_for_txt(m.txt_path)

    keys = set(py_map.keys()) | set(double_map.keys())
    pairs: List[Tuple[TrialKey, Optional[TrialMetrics], Optional[TrialMetrics]]] = []
    for hemi, target in sorted(keys):
        pairs.append((TrialKey(hemisphere=hemi, target=target), py_map.get((hemi, target)), double_map.get((hemi, target))))
    return pairs


def _plot_pair(key: TrialKey, py: Optional[TrialMetrics], dbl: Optional[TrialMetrics]) -> plt.Figure:
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

    ax_img_py = fig.add_subplot(gs[0, 0])
    ax_img_dbl = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    fig.suptitle(f"{key.hemisphere} - {key.target}")

    # Images
    def _show(ax, tm: Optional[TrialMetrics], title: str):
        if tm and tm.img_path and os.path.exists(tm.img_path):
            img = imread(tm.img_path)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{title}\n{os.path.basename(tm.img_path)}", fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center')
            ax.set_title(title)
        ax.axis('off')

    _show(ax_img_py, py, 'Single (py)')
    _show(ax_img_dbl, dbl, 'Dual (double)')
    print(f"[DEBUG] Key: {key.hemisphere} {key.target}")
    if py:
        print(f"[DEBUG] py img: {py.img_path} exists={os.path.exists(py.img_path) if py.img_path else False}")
    if dbl:
        print(f"[DEBUG] dbl img: {dbl.img_path} exists={os.path.exists(dbl.img_path) if dbl.img_path else False}")

    # Metrics bar chart
    labels = ['Dice', 'Yield', 'Hellinger (lower=better)', 'Loss']
    x = np.arange(len(labels))
    width = 0.35
    py_vals = [np.nan, np.nan, np.nan, np.nan]
    dbl_vals = [np.nan, np.nan, np.nan, np.nan]
    if py:
        py_loss = (1 - LOSS_A * py.dice) + (1 - LOSS_B * py.yield_) + (LOSS_C * py.hell)
        py_vals = [py.dice, py.yield_, py.hell, py_loss]
        print(f"[DEBUG] py vals: {py_vals}")
    if dbl:
        dbl_loss = (1 - LOSS_A * dbl.dice) + (1 - LOSS_B * dbl.yield_) + (LOSS_C * dbl.hell)
        dbl_vals = [dbl.dice, dbl.yield_, dbl.hell, dbl_loss]
        print(f"[DEBUG] dbl vals: {dbl_vals}")

    # Replace NaN with 0 for plotting, but keep a mask for label display
    py_mask = np.isnan(py_vals)
    dbl_mask = np.isnan(dbl_vals)
    py_plot = np.where(py_mask, 0.0, py_vals)
    dbl_plot = np.where(dbl_mask, 0.0, dbl_vals)

    bars_py = ax_bar.bar(x - width/2, py_plot, width, label='py')
    bars_dbl = ax_bar.bar(x + width/2, dbl_plot, width, label='double')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    # Annotate values
    def _annotate(bars, vals):
        for rect, v in zip(bars, vals):
            if np.isnan(v):
                continue
            height = rect.get_height()
            ax_bar.text(rect.get_x() + rect.get_width()/2, height,
                        f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    _annotate(bars_py, py_vals)
    _annotate(bars_dbl, dbl_vals)

    ax_bar.legend()
    ax_bar.grid(True, axis='y', alpha=0.3)

    # Set sensible y-limit
    ymax = np.nanmax([py_vals, dbl_vals])
    if np.isfinite(ymax):
        ax_bar.set_ylim(0, max(1.0, ymax * 1.2))

    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare single (py) vs double outputs for 102311')
    parser.add_argument('--py_txt', type=str, default=None, help='Path to single (py) txt file')
    parser.add_argument('--double_txt', type=str, default=None, help='Path to double txt file')
    args = parser.parse_args()
    os.makedirs(OUT_DOUBLE, exist_ok=True)
    # Remove existing compare images
    for p in glob.glob(os.path.join(OUT_DOUBLE, 'compare_*.png')):
        try:
            os.remove(p)
        except Exception:
            pass
    # Direct compare mode
    if args.py_txt and args.double_txt:
        py_tm = _parse_metrics_txt(args.py_txt)
        dbl_tm = _parse_metrics_txt(args.double_txt)
        if not py_tm or not dbl_tm:
            print('Failed to parse one of the provided txt files.')
            return
        # Attach images
        def _img_for_txt(txt_path: str) -> Optional[str]:
            base = os.path.splitext(txt_path)[0]
            img_path = base + _IMG_SUFFIX
            return img_path if os.path.exists(img_path) else None
        py_tm.img_path = _img_for_txt(py_tm.txt_path)
        dbl_tm.img_path = _img_for_txt(dbl_tm.txt_path)
        # Build key from any one (fallback if regex fails)
        key = _build_key_from_filename(args.py_txt) or TrialKey(hemisphere='LH', target='targ-full')
        print(f"[DEBUG] Direct mode key: {key}")
        fig = _plot_pair(key, py_tm, dbl_tm)
        out_name = f"compare_{key.hemisphere}_{key.target}.png"
        out_path = os.path.join(OUT_DOUBLE, out_name)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', out_path)
        return

    # Batch mode
    pairs = _pair_trials()
    print(f"[DEBUG] Found pairs: {len(pairs)}")
    if not pairs:
        print('No pairs found to compare.')
        return

    for key, py, dbl in pairs:
        print(f"[DEBUG] Pair -> {key}")
        fig = _plot_pair(key, py, dbl)
        out_name = f"compare_{key.hemisphere}_{key.target}.png"
        out_path = os.path.join(OUT_DOUBLE, out_name)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', out_path)


if __name__ == '__main__':
    main()


