# -*- coding: utf-8 -*-
"""
0.5/1.0/1.5mm 스페이싱으로 최적화된 결과(PKL)를 입력으로 받아,
각 스페이싱에 대해 그리디 방식으로 채널 부분집합을 선택하고 결과를 저장/비교합니다.

입력: data/output/100610_spacing/spacing_{spacing}_{grid}/ 내 *.pkl
출력: data/1028_greedy_100610_spacing/{spacing}/{HEM}_{target}/ 하위에 저장
"""

import os
import sys
import csv
import pickle
import itertools
import time
from typing import List, Dict, Tuple, Optional

# CPU 사용률 제한 설정 (80%)
CPU_LIMIT = 0.8  # 80% 사용률로 제한

# NumPy 스레드 수 제한 (CPU 코어 수의 80% 사용)
try:
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # NumPy가 사용할 수 있는 스레드 수를 CPU 코어 수의 80%로 제한
    numpy_threads = max(1, int(cpu_count * CPU_LIMIT))
    os.environ['OMP_NUM_THREADS'] = str(numpy_threads)
    os.environ['MKL_NUM_THREADS'] = str(numpy_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(numpy_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(numpy_threads)
    print(f"[CPU] CPU 코어 수: {cpu_count}, NumPy 스레드 수 제한: {numpy_threads} (80%)")
except Exception as e:
    print(f"[WARN] CPU 스레드 제한 설정 실패: {e}")

# code 폴더를 모듈 경로에 추가
# 파일 위치: code/greedy/bayesianopt_V1_1110_greedy.py
# PROJ_ROOT는 프로젝트 루트 (vimplant0812 폴더)
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = os.path.join(PROJ_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np
import matplotlib.pyplot as plt

# CPU 사용률 모니터링 (선택사항, psutil이 있으면 사용)
try:
    import psutil
    HAS_PSUTIL = True
    print(f"[CPU] psutil 사용 가능 - CPU 사용률 모니터링 활성화")
except ImportError:
    HAS_PSUTIL = False
    print(f"[CPU] psutil 없음 - NumPy 스레드 제한만 적용")

from lossfunc import DC, hellinger_distance, get_yield
from electphos import prf_to_phos
import visualsectors as gvs


# 경로 상수 (위에서 이미 계산됨, 필요시 수정)
# PROJ_ROOT는 위에서 동적으로 계산됨
SPACING_BASE_INPUT = os.path.join(PROJ_ROOT, "data", "output", "100610_spacing")
SPACING_BASE_OUTPUT = os.path.join(PROJ_ROOT, "data", "1028_greedy_100610_spacing")

# 스페이싱 목록
SPACING_NAMES = ["0.5mm", "1.0mm", "1.5mm"]

# 고정 설정 (원 로직과 일치)
WINDOWSIZE = 1000
VIEW_ANGLE = 90
DC_PERCENTILE = 50
WEIGHT_A = 1.0   # dice 가중치
WEIGHT_B = 0.1   # yield 가중치
WEIGHT_C = 1.0   # hellinger 가중치
PENALTY = 0.25   # grid invalid penalty
VERBOSE = True


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def normalize_probability_map(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    maxv = float(out.max()) if out.size > 0 else 0.0
    if maxv > 0:
        out /= maxv
    sumv = float(out.sum())
    if sumv > 0:
        out /= sumv
    return out


def render_map_from_subset(phosphenes_v1: np.ndarray, active_idx: List[int]) -> np.ndarray:
    """
    저장된 phosphenes_V1는 이미 sigma가 포함되어 있다는 가정으로,
    활성 채널 subset만 사용하여 raw phosphene map을 합성 후 확률맵으로 정규화.
    """
    canvas = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype=np.float32)
    if not active_idx:
        return canvas
    subset = phosphenes_v1[np.array(active_idx, dtype=int)]
    canvas = prf_to_phos(canvas, subset, view_angle=VIEW_ANGLE, phSizeScale=1)
    return normalize_probability_map(canvas)


def subset_contacts(contacts_xyz_moved: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    contacts_xyz_moved의 저장 방향(3xN 또는 Nx3)에 맞춰 인덱스 목록으로 부분집합을 반환.
    """
    arr = np.asarray(contacts_xyz_moved)
    if len(indices) == 0:
        if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
            return arr[:, []]
        else:
            return arr[[]]
    idx = np.array(indices, dtype=int)
    if arr.ndim == 2 and arr.shape[0] == 3 and (arr.shape[1] != 3):
        return arr[:, idx]
    else:
        return arr[idx]


def compute_loss(target_density: np.ndarray,
                 phos_map: np.ndarray,
                 contacts_subset: np.ndarray,
                 good_coords: np.ndarray,
                 grid_valid: bool,
                 a: float = WEIGHT_A,
                 b: float = WEIGHT_B,
                 c: float = WEIGHT_C) -> Tuple[float, Dict[str, float]]:
    """
    원 비용식: cost = (1 - a*dice) + (1 - b*yield) + c*hellinger, grid invalid 시 항마다 PENALTY.
    """
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
    if not grid_valid:
        cost = par1 + PENALTY + par2 + PENALTY + par3 + PENALTY

    if np.isnan(cost) or np.isinf(cost):
        cost = 3.0
    return float(cost), {"dice": float(dice), "hell": float(hell), "yield": float(grid_yield)}


def _check_and_limit_cpu():
    """CPU 사용률을 체크하고 80%를 초과하면 대기"""
    if not HAS_PSUTIL:
        return
    
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > CPU_LIMIT * 100:
            # CPU 사용률이 80%를 초과하면 잠시 대기
            sleep_time = (cpu_percent - CPU_LIMIT * 100) / 100.0 * 0.1
            time.sleep(min(sleep_time, 0.1))  # 최대 0.1초 대기
    except Exception:
        pass  # 모니터링 실패해도 계속 진행


def greedy_select(phosphenes_v1: np.ndarray,
                  contacts_xyz_moved: np.ndarray,
                  good_coords: np.ndarray,
                  grid_valid: bool,
                  target_density: np.ndarray,
                  min_improvement: float = 1e-6) -> Tuple[List[int], np.ndarray, List[float], List[Dict]]:
    """
    채널을 하나씩 추가하며 비용이 감소하는 동안 계속 진행. 개선이 min_improvement 이하이면 중단.
    50개까지는 무조건 진행, 그 이후 개선 없으면 중단.
    CPU 사용률을 80%로 제한합니다.
    """
    n = int(phosphenes_v1.shape[0])
    remaining = list(range(n))
    active = []

    current_map = render_map_from_subset(phosphenes_v1, active)
    contacts_subset = subset_contacts(contacts_xyz_moved, active)
    best_loss, comps0 = compute_loss(target_density, current_map, contacts_subset, good_coords, grid_valid)
    history = [best_loss]
    progress = []

    step = 0
    while remaining:
        # CPU 사용률 체크 및 제한
        _check_and_limit_cpu()
        best_i = None
        best_trial_loss = None
        best_comps = None

        for i in remaining:
            # CPU 사용률 체크 (매 10개 채널마다)
            if len(remaining) % 10 == 0:
                _check_and_limit_cpu()
            
            trial_active = active + [i]
            trial_map = render_map_from_subset(phosphenes_v1, trial_active)
            contacts_subset = subset_contacts(contacts_xyz_moved, trial_active)
            trial_loss, comps = compute_loss(target_density, trial_map, contacts_subset, good_coords, grid_valid)
            if (best_trial_loss is None) or (trial_loss < best_trial_loss):
                best_trial_loss = trial_loss
                best_i = i
                best_comps = comps
            if VERBOSE:
                delta = best_loss - trial_loss
                print(f"trial step={step} cand={i} loss={trial_loss:.6f} delta={delta:.6g} dice={comps['dice']:.4f} hell={comps['hell']:.4f} yield={comps['yield']:.4f}")

        delta_best = best_loss - best_trial_loss if best_trial_loss is not None else 0.0
        if len(active) >= 50 and delta_best <= min_improvement:
            if VERBOSE:
                print(f"stop at step={step}: no improvement after 50 (delta={delta_best:.6g})")
            break

        active.append(best_i)               # 선택
        remaining.remove(best_i)
        current_map = render_map_from_subset(phosphenes_v1, active)
        contacts_subset = subset_contacts(contacts_xyz_moved, active)
        best_loss, comps = compute_loss(target_density, current_map, contacts_subset, good_coords, grid_valid)
        history.append(best_loss)
        progress.append({
            "step": step,
            "picked_index": best_i,
            "active_count": len(active),
            "loss": best_loss,
            "dice": comps['dice'],
            "hell": comps['hell'],
            "yield": comps['yield']
        })
        if VERBOSE:
            print(f"accept step={step} pick={best_i} new_loss={best_loss:.6f} dice={comps['dice']:.4f} hell={comps['hell']:.4f} yield={comps['yield']:.4f}")
        step += 1

    return active, current_map, history, progress


def _parse_hem_and_slug(pkl_name: str) -> Tuple[str, str]:
    hem = "LH" if "_LH_" in pkl_name else ("RH" if "_RH_" in pkl_name else "HEM")
    target_slug = "full"
    if "targ-" in pkl_name:
        try:
            target_slug = pkl_name.split("targ-")[-1].rsplit(".", 1)[0]
        except Exception:
            target_slug = "full"
    return hem, target_slug


def _load_processed_target_by_slug(slug: str) -> Optional[np.ndarray]:
    base_rt = os.path.join(PROJ_ROOT, "new_targets", "processed_targets")
    npy_path = os.path.join(base_rt, f"{slug}.npy")
    if os.path.isfile(npy_path):
        try:
            arr = np.load(npy_path)
            return normalize_probability_map(arr.astype(np.float64))
        except Exception:
            return None
    return None


def build_full_target() -> np.ndarray:
    t = gvs.complete_gauss(
        windowsize=WINDOWSIZE,
        fwhm=1200,
        radiusLow=0,
        radiusHigh=500,
        center=None,
        plotting=False
    )
    return normalize_probability_map(t.astype(np.float64))


def build_target_by_slug(slug: str) -> np.ndarray:
    if slug == "full":
        return build_full_target()
    if slug == "upper":
        t = gvs.upper_sector(windowsize=WINDOWSIZE, fwhm=400, radiusLow=0, radiusHigh=250, plotting=False)
        return normalize_probability_map(t.astype(np.float64))
    if slug == "lower":
        t = gvs.lower_sector(windowsize=WINDOWSIZE, fwhm=400, radiusLow=0, radiusHigh=250, plotting=False)
        return normalize_probability_map(t.astype(np.float64))
    if slug == "inner":
        t = gvs.inner_ring(windowsize=WINDOWSIZE, fwhm=400, radiusLow=0, radiusHigh=250, plotting=False)
        return normalize_probability_map(t.astype(np.float64))
    loaded = _load_processed_target_by_slug(slug)
    if loaded is not None:
        return loaded
    return build_full_target()


def _find_spacing_pkl_files(spacing_name: str) -> List[str]:
    """
    data/output/100610_spacing/ 아래에서 spacing_{spacing_name}_* 하위의 *.pkl 수집
    """
    result = []
    if not os.path.isdir(SPACING_BASE_INPUT):
        return result
    for entry in os.listdir(SPACING_BASE_INPUT):
        if not entry.startswith(f"spacing_{spacing_name}_"):
            continue
        subdir = os.path.join(SPACING_BASE_INPUT, entry)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.endswith(".pkl") and ("_LH_" in fname or "_RH_" in fname) and "targ-" in fname:
                result.append(os.path.join(subdir, fname))
    # 최신 우선 정렬(수정시간 역순)
    result.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return result


def _process_one_spacing_pkl(pkl_path: str, spacing_name: str, output_base: str):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    pkl_name = os.path.basename(pkl_path)
    hem, target_slug = _parse_hem_and_slug(pkl_name)
    out_dir = os.path.join(output_base, spacing_name, f"{hem}_{target_slug}")
    # 이미 결과가 일부라도 있으면 스킵(덮어쓰기 원하면 조건 제거)
    if os.path.isdir(out_dir):
        try:
            with os.scandir(out_dir) as it:
                for _ in it:
                    print(f"[INFO] Skip exists: {spacing_name}/{hem}_{target_slug}")
                    return
        except Exception:
            pass
    ensure_dir(out_dir)

    # 인덱스 매핑 (1013_spacing 저장 구조와 동일)
    grid_valid = bool(payload[1])
    contacts_xyz_moved = payload[5]
    good_coords = payload[6]
    phosphenes_v1 = payload[11]

    # 타깃 생성/로드
    target_density = build_target_by_slug(target_slug)

    # 그리디 채널 선택
    active_indices, final_map, loss_history, progress = greedy_select(
        phosphenes_v1,
        contacts_xyz_moved,
        good_coords,
        grid_valid,
        target_density
    )

    # 전체 채널 맵
    all_indices = list(range(int(phosphenes_v1.shape[0])))
    full_map = render_map_from_subset(phosphenes_v1, all_indices)

    # 저장 (npy)
    np.save(os.path.join(out_dir, "selected_indices.npy"), np.array(active_indices, dtype=np.int32))
    np.save(os.path.join(out_dir, "final_phosphene_map.npy"), final_map.astype(np.float32))
    np.save(os.path.join(out_dir, "full_phosphene_map.npy"), full_map.astype(np.float32))

    # 요약 텍스트
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as wf:
        wf.write(f"spacing: {spacing_name}\n")
        wf.write(f"pkl: {pkl_name}\n")
        wf.write(f"hemisphere: {hem}\n")
        wf.write(f"target_slug: {target_slug}\n")
        wf.write(f"selected_count: {len(active_indices)}\n")
        wf.write(f"final_loss: {loss_history[-1] if loss_history else np.nan}\n")
        wf.write(f"loss_history_len: {len(loss_history)}\n")
        wf.write("indices_order:\n")
        wf.write(",".join(map(str, active_indices)))

    # Raw phosphene map images (selected vs all)
    plt.figure(figsize=(6,5))
    plt.imshow(final_map, cmap='seismic', vmin=0, vmax=float(np.max(final_map)) if np.max(final_map) > 0 else 1)
    plt.title('Selected channels - raw phosphene map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "selected_raw_phosphene_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(full_map, cmap='seismic', vmin=0, vmax=float(np.max(full_map)) if np.max(full_map) > 0 else 1)
    plt.title('All channels - raw phosphene map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "allchannels_raw_phosphene_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 비교 이미지: Target vs Selected vs Full
    plt.figure(figsize=(15, 5))
    vmax = float(max(np.max(target_density), np.max(final_map), np.max(full_map)))
    vmax = vmax if vmax > 0 else 1.0
    plt.subplot(1,3,1)
    plt.imshow(target_density, cmap='seismic', vmin=0, vmax=vmax)
    plt.title('Target density (raw)')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(final_map, cmap='seismic', vmin=0, vmax=vmax)
    plt.title('Selected channels (raw)')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(full_map, cmap='seismic', vmin=0, vmax=vmax)
    plt.title('All channels (raw)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_target_selected_full.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 진행 상황 CSV 저장
    csv_path = os.path.join(out_dir, "greedy_progress.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["step", "picked_index", "active_count", "loss", "dice", "hell", "yield"]) 
        for row in progress:
            writer.writerow([row["step"], row["picked_index"], row["active_count"],
                             f"{row['loss']:.6f}", f"{row['dice']:.6f}", f"{row['hell']:.6f}", f"{row['yield']:.6f}"])

    print("Saved:")
    print(" -", os.path.join(out_dir, "selected_indices.npy"))
    print(" -", os.path.join(out_dir, "final_phosphene_map.npy"))
    print(" -", os.path.join(out_dir, "full_phosphene_map.npy"))
    print(" -", os.path.join(out_dir, "summary.txt"))
    print(" -", os.path.join(out_dir, "selected_raw_phosphene_map.png"))
    print(" -", os.path.join(out_dir, "allchannels_raw_phosphene_map.png"))
    print(" -", os.path.join(out_dir, "comparison_target_selected_full.png"))
    print(" -", os.path.join(out_dir, "greedy_progress.csv"))


def _gather_results_for_comparison(output_base: str) -> Dict[str, Dict[str, Dict]]:
    """
    비교용으로 결과를 모읍니다.
    구조: grouped[hemi_target][spacing_name] = { 'final_map', 'contacts_xyz_moved'(없음), 'summary metrics' ... }
    여기서는 final_map만으로 비교 플롯을 생성하고, 필요시 summary.txt 파싱 가능.
    """
    grouped: Dict[str, Dict[str, Dict]] = {}
    for spacing in SPACING_NAMES:
        spacing_dir = os.path.join(output_base, spacing)
        if not os.path.isdir(spacing_dir):
            continue
        for sub in os.listdir(spacing_dir):
            subdir = os.path.join(spacing_dir, sub)
            if not os.path.isdir(subdir):
                continue
            hemi_target = sub  # "{HEM}_{target_slug}"
            final_map_path = os.path.join(subdir, "final_phosphene_map.npy")
            if not os.path.isfile(final_map_path):
                continue
            try:
                final_map = np.load(final_map_path)
            except Exception:
                continue
            grouped.setdefault(hemi_target, {})[spacing] = {
                "final_map": final_map
            }
    return grouped


def _plot_triple_comparison(grouped: Dict[str, Dict[str, Dict]], output_base: str):
    """
    같은 HEM+타깃에 대해 0.5/1.0/1.5mm 결과를 한 장(1행 3열)으로 비교 저장.
    """
    for hemi_target, by_spacing in grouped.items():
        ordered = [by_spacing.get(s) for s in SPACING_NAMES if by_spacing.get(s) is not None]
        if len(ordered) != 3:
            continue
        # 공통 vmax
        vmax = max(float(np.max(r["final_map"])) for r in ordered)
        vmax = vmax if vmax > 0 else 1.0
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{hemi_target} | Greedy Final Map (0.5/1.0/1.5mm)")
        for i, spacing in enumerate(SPACING_NAMES):
            r = by_spacing.get(spacing)
            ax = axes[i]
            ax.imshow(r["final_map"], cmap='seismic', vmin=0, vmax=vmax)
            ax.set_title(spacing)
            ax.axis('off')
        plt.tight_layout()
        out_dir = os.path.join(output_base, "comparison")
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"greedy_finalmap_triple_{hemi_target}_0.5_1.0_1.5.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[COMPARE] saved {out_path}")


def main():
    ensure_dir(SPACING_BASE_OUTPUT)
    
    # CPU 사용률 제한 정보 출력
    if HAS_PSUTIL:
        print(f"[CPU] CPU 사용률 모니터링 활성화 (목표: {CPU_LIMIT*100:.0f}%)")
    else:
        print(f"[CPU] NumPy 스레드 제한만 적용 (목표: {CPU_LIMIT*100:.0f}%)")
        print(f"[CPU] psutil 설치 시 더 정확한 CPU 사용률 제한 가능: pip install psutil")

    # 1) 스페이싱별 PKL 수집 및 처리
    for spacing_name in SPACING_NAMES:
        print(f"\n{'='*60}\n[SPACING] {spacing_name}\n{'='*60}")
        pkl_files = _find_spacing_pkl_files(spacing_name)
        if not pkl_files:
            print(f"[WARN] No PKL found for spacing {spacing_name}")
            continue
        # 동일 HEM/타깃의 중복이 여럿 있을 수 있어 최신 파일부터 처리(이미 결과 있으면 스킵)
        for pkl_path in pkl_files:
            try:
                print(f"[INFO] Processing: {os.path.basename(pkl_path)}")
                # CPU 사용률 체크
                _check_and_limit_cpu()
                _process_one_spacing_pkl(pkl_path, spacing_name, SPACING_BASE_OUTPUT)
            except Exception as e:
                print(f"[WARN] Skip {os.path.basename(pkl_path)} due to error: {e}")

    # 2) 비교 플롯 생성 (최종 맵 기준의 간단 비교)
    grouped = _gather_results_for_comparison(SPACING_BASE_OUTPUT)
    _plot_triple_comparison(grouped, SPACING_BASE_OUTPUT)


if __name__ == "__main__":
    main()