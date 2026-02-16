# -*- coding: utf-8 -*-
"""
100610 LH full 타겟에 대해, 이미 계산된 최적 파라미터/임플란트 결과(PKL)를 사용하여
채널을 하나씩 그리디로 켜며 원래 로스 공식을 그대로 적용해 비용을 최소화한다.
감소가 멈추면 선택을 중단하고 다음을 저장한다.

저장물 (data/1028_greedy_100610):
- selected_indices.npy: 선택된 채널 인덱스 순서
- final_phosphene_map.npy: 선택된 채널 raw 맵 (정규화된 확률 맵)
- full_phosphene_map.npy: 모든 채널 raw 맵
- selected_raw_phosphene_map.png, allchannels_raw_phosphene_map.png: 시각화
- comparison_target_selected_full.png: 타겟/선택/전체 비교 이미지
- greedy_progress.csv: 단계별 선택, loss, dice, hell, yield
- summary.txt: 요약
"""

import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lossfunc import DC, hellinger_distance, get_yield
from electphos import prf_to_phos
import visualsectors as gvs


# 고정 설정 (원 코드와 일치)
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


def render_map_from_subset(phosphenes_v1: np.ndarray, active_idx: list) -> np.ndarray:
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


def subset_contacts(contacts_xyz_moved: np.ndarray, indices: list) -> np.ndarray:
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
                 c: float = WEIGHT_C):
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


def greedy_select(phosphenes_v1: np.ndarray,
                  contacts_xyz_moved: np.ndarray,
                  good_coords: np.ndarray,
                  grid_valid: bool,
                  target_density: np.ndarray,
                  min_improvement: float = 1e-6):
    """
    채널을 하나씩 추가하며 비용이 감소하는 동안 계속 진행. 개선이 min_improvement 이하이면 중단.
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
        # argmin 전략: 현재 active에 하나를 추가했을 때 trial loss가 최소가 되는 채널 선택
        best_i = None
        best_trial_loss = None
        best_comps = None

        for i in remaining:
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

        # 50개까지는 무조건 진행, 그 이후에는 개선 없으면 중단
        delta_best = best_loss - best_trial_loss if best_trial_loss is not None else 0.0
        if len(active) >= 50 and delta_best <= min_improvement:
            if VERBOSE:
                print(f"stop at step={step}: no improvement after 50 (delta={delta_best:.6g})")
            break

        # 선택 및 갱신
        active.append(best_i)
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


def load_lh_full_pkl(base_dir: str) -> str:
    fname = "100610_LH_V1_n1000_1x10_dice-yield-HD_0.05_targ-full.pkl"
    fpath = os.path.join(base_dir, fname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Not found: {fpath}")
    return fpath


def build_full_target() -> np.ndarray:
    target = gvs.complete_gauss(
        windowsize=WINDOWSIZE,
        fwhm=1200,
        radiusLow=0,
        radiusHigh=500,
        center=None,
        plotting=False
    )
    return normalize_probability_map(target.astype(np.float64))


def _parse_hem_and_slug(pkl_name: str):
    hem = "LH" if "_LH_" in pkl_name else ("RH" if "_RH_" in pkl_name else "HEM")
    target_slug = "full"
    if "targ-" in pkl_name:
        try:
            target_slug = pkl_name.split("targ-")[-1].rsplit(".", 1)[0]
        except Exception:
            target_slug = "full"
    return hem, target_slug


def _load_processed_target_by_slug(slug: str):
    base_rt = os.path.join("C:/Users/user/YongtaeC/vimplant0812", "new_targets", "processed_targets")
    npy_path = os.path.join(base_rt, f"{slug}.npy")
    if os.path.isfile(npy_path):
        try:
            arr = np.load(npy_path)
            return normalize_probability_map(arr.astype(np.float64))
        except Exception:
            return None
    return None


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
    # 시도: 전처리된 타겟 npy 로드 (letters 등)
    loaded = _load_processed_target_by_slug(slug)
    if loaded is not None:
        return loaded
    # 마지막 수단: full 반환
    return build_full_target()


def _process_one(pkl_path: str, output_base: str):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    pkl_name = os.path.basename(pkl_path)
    hem, target_slug = _parse_hem_and_slug(pkl_name)
    subdir = f"{hem}_full" if target_slug == "full" else f"{hem}_{target_slug}"
    output_dir = os.path.join(output_base, subdir)
    # 이미 해당 서브폴더에 결과 파일이 존재하면 스킵
    if os.path.isdir(output_dir):
        try:
            has_any = False
            with os.scandir(output_dir) as it:
                for _ in it:
                    has_any = True
                    break
            if has_any:
                print(f"[INFO] Skip (already exists): {subdir}")
                return
        except Exception:
            # 폴더 접근 문제 등은 계속 진행해서 새로 생성
            pass
    ensure_dir(output_dir)

    # 저장 포맷 인덱스
    grid_valid = bool(payload[1])
    contacts_xyz_moved = payload[5]
    good_coords = payload[6]
    phosphenes_v1 = payload[11]

    # 타겟 생성/로드
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
    np.save(os.path.join(output_dir, "selected_indices.npy"), np.array(active_indices, dtype=np.int32))
    np.save(os.path.join(output_dir, "final_phosphene_map.npy"), final_map.astype(np.float32))
    np.save(os.path.join(output_dir, "full_phosphene_map.npy"), full_map.astype(np.float32))

    # 요약 텍스트
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as wf:
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
    plt.savefig(os.path.join(output_dir, "selected_raw_phosphene_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(full_map, cmap='seismic', vmin=0, vmax=float(np.max(full_map)) if np.max(full_map) > 0 else 1)
    plt.title('All channels - raw phosphene map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "allchannels_raw_phosphene_map.png"), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, "comparison_target_selected_full.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 진행 상황 CSV 저장
    csv_path = os.path.join(output_dir, "greedy_progress.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["step", "picked_index", "active_count", "loss", "dice", "hell", "yield"]) 
        for row in progress:
            writer.writerow([row["step"], row["picked_index"], row["active_count"],
                             f"{row['loss']:.6f}", f"{row['dice']:.6f}", f"{row['hell']:.6f}", f"{row['yield']:.6f}"])

    # 메트릭 곡선 저장 (x축=채널 개수)
    counts = [r["active_count"] for r in progress]
    losses = [r["loss"] for r in progress]
    dices = [r["dice"] for r in progress]
    hells = [r["hell"] for r in progress]
    yields = [r["yield"] for r in progress]

    plt.figure(figsize=(10,6))
    plt.plot(counts, losses, label='loss')
    plt.plot(counts, dices, label='dice')
    plt.plot(counts, hells, label='hellinger')
    plt.plot(counts, yields, label='yield')
    plt.xlabel('number of active channels')
    plt.ylabel('metric value')
    plt.title('Metrics vs number of active channels (greedy order)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_vs_active_count.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 간단 출력
    print("Saved:")
    print(" -", os.path.join(output_dir, "selected_indices.npy"))
    print(" -", os.path.join(output_dir, "final_phosphene_map.npy"))
    print(" -", os.path.join(output_dir, "full_phosphene_map.npy"))
    print(" -", os.path.join(output_dir, "summary.txt"))
    print(" -", os.path.join(output_dir, "selected_raw_phosphene_map.png"))
    print(" -", os.path.join(output_dir, "allchannels_raw_phosphene_map.png"))
    print(" -", os.path.join(output_dir, "comparison_target_selected_full.png"))
    print(" -", os.path.join(output_dir, "greedy_progress.csv"))


def main():
    input_dir = os.path.join("C:/Users/user/YongtaeC/vimplant0812", "data", "0920_new_targets_100610")
    output_base = os.path.join("C:/Users/user/YongtaeC/vimplant0812", "data", "1028_greedy_100610")
    ensure_dir(output_base)

    # 입력 폴더의 모든 LH/RH 타겟 PKL 순회
    pkl_files = []
    for fname in os.listdir(input_dir):
        if fname.endswith('.pkl') and ('_LH_' in fname or '_RH_' in fname) and 'targ-' in fname:
            pkl_files.append(os.path.join(input_dir, fname))
    pkl_files.sort()

    if not pkl_files:
        # 기존 단일 LH_full 파일 시도 (호환)
        try:
            single = load_lh_full_pkl(input_dir)
            pkl_files = [single]
        except Exception:
            raise FileNotFoundError(f"입력 폴더에 처리할 PKL이 없습니다: {input_dir}")

    for pkl_path in pkl_files:
        try:
            print(f"[INFO] Processing: {os.path.basename(pkl_path)}")
            _process_one(pkl_path, output_base)
        except Exception as e:
            print(f"[WARN] Skip {os.path.basename(pkl_path)} due to error: {e}")


if __name__ == "__main__":
    main()
