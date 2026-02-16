# -*- coding: utf-8 -*-
"""
다중 전극(2개) 최적화: 1x1x10(깊이) 전극 두 개를 동시에 최적화
- 0920(단일) + 0828_double(이중) 코드를 바탕으로 구성
- 두 전극의 컨택트를 합쳐 하나의 포스핀 맵을 만들고 타깃과 비교하여 단일 코스트 최소화
"""

import time
import os.path
import pickle
from copy import deepcopy
from typing import Any
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skopt.utils import use_named_args
from skopt.space import Integer
from skopt.utils import cook_initial_point_generator
from skopt import gp_minimize

########################
### Custom functions ###
########################
from ninimplant import get_xyz
from lossfunc import DC, get_yield, hellinger_distance
from electphos import create_grid_v2, implant_grid, get_phosphenes, prf_to_phos, get_cortical_magnification, cortical_spread
from scipy.spatial import cKDTree
import visualsectors as gvs

# ignore "True-divide" warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.seterr(divide='ignore', invalid='ignore')

##########
## INIT ##
##########

# 데이터 경로
datafolder = "C:/Users/user/YongtaeC/vimplant0812/data/input/100610/"
outputfolder = "C:/Users/user/YongtaeC/vimplant0812/data/1110_multi_electrode_100610/"
os.makedirs(outputfolder, exist_ok=True)

# 최적화 변수: 전극 A(4) + 전극 B(4) = 8차원
dimA1 = Integer(name='alpha_A', low=-90, high=90)      # deg
dimA2 = Integer(name='beta_A',  low=-15, high=110)     # LH 기본, RH는 루프에서 교체
dimA3 = Integer(name='offset_from_base_A', low=0, high=40)  # mm
dimA4 = Integer(name='shank_length_A',     low=10, high=40) # mm

dimB1 = Integer(name='alpha_B', low=-90, high=90)
dimB2 = Integer(name='beta_B',  low=-15, high=110)
dimB3 = Integer(name='offset_from_base_B', low=0, high=40)
dimB4 = Integer(name='shank_length_B',     low=10, high=40)

dimensions_default = [dimA1, dimA2, dimA3, dimA4, dimB1, dimB2, dimB3, dimB4]

num_calls = 150
x0 = (0,0,20,25, 0,0,20,25)
num_initial_points = 10
dc_percentile = 50
WINDOWSIZE = 1000

# 5x10x10(깊이) 전극 설정
grid_nx  = 5
grid_ny  = 10
grid_nz  = 10
spacing_x = 1.0   # mm
spacing_y = 1.0   # mm

# 두 번째 전극도 동일 설정
grid2_nx  = 5
grid2_ny  = 10
grid2_nz  = 10
spacing2_x = spacing_x
spacing2_y = spacing_y

# 두 전극 시작 위치 분리(초기 오프셋)
start_location_B_offset = [0.0, spacing_y * grid_ny, 0.0]   # mm

# 두 전극 최소 거리(mm) 제약
min_interprobe_distance_mm = max(spacing_x * grid_nx, spacing_y * grid_ny)

def _min_interprobe_distance_mm(contacts_A, contacts_B):
    if contacts_A.size == 0 or contacts_B.size == 0:
        return np.inf
    treeA = cKDTree(contacts_A.T)
    dists, _ = treeA.query(contacts_B.T, k=1)
    return float(np.min(dists))

# 손실 가중치 및 타깃
loss_comb  = ([(1, 0.1, 1)])   # (a, b, c) = (Dice, Yield, Hellinger)
loss_names = (['dice-yield-HD'])

targ_comb = ([
    gvs.upper_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False),
    gvs.lower_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False),
    gvs.inner_ring(windowsize=WINDOWSIZE, fwhm=400, radiusLow=0, radiusHigh=250, plotting=False),
    gvs.complete_gauss(windowsize=1000, fwhm=1200, radiusLow=0, radiusHigh=500, center=None, plotting=False)
])
targ_names = (['targ-upper', 'targ-lower', 'targ-inner', 'targ-full'])

# pRF 모델 상수
cort_mag_model = 'wedge-dipole'
view_angle = 90
amp = 100

# 스토퍼
def custom_stopper(res, N=5, delta=0.2, thresh=0.05):
    if len(res.func_vals) >= N:
        func_vals = np.sort(res.func_vals)
        worst = func_vals[N - 1]
        best = func_vals[0]
        return (abs((best - worst)/worst) < delta) & (best < thresh)
    else:
        return None

# 수동 재현 및 저장용
def f_manual(alpha_A, beta_A, offset_from_base_A, shank_length_A,
             alpha_B, beta_B, offset_from_base_B, shank_length_B,
             good_coords, good_coords_V1, good_coords_V2, good_coords_V3,
             target_density,
             start_location, gm_mask, polar_map, ecc_map, sigma_map,
             a, b, c):
    penalty = 0.25
    new_angle_A = (float(alpha_A), float(beta_A), 0)
    new_angle_B = (float(alpha_B), float(beta_B), 0)

    # Grid A 생성
    orig_grid_A = create_grid_v2(
        start_location,
        shank_length=shank_length_A,
        n_x=grid_nx, n_y=grid_ny, n_z=grid_nz,
        spacing_x=spacing_x, spacing_y=spacing_y,
        offset_from_origin=0
    )
    # Grid B 생성 (시작점 오프셋)
    start_location_B = [
        start_location[0] + start_location_B_offset[0],
        start_location[1] + start_location_B_offset[1],
        start_location[2] + start_location_B_offset[2],
    ]
    orig_grid_B = create_grid_v2(
        start_location_B,
        shank_length=shank_length_B,
        n_x=grid2_nx, n_y=grid2_ny, n_z=grid2_nz,
        spacing_x=spacing2_x, spacing_y=spacing2_y,
        offset_from_origin=0
    )

    # Implant
    _, contacts_xyz_moved_A, _, _, _, _, _, _, grid_valid_A = implant_grid(gm_mask, orig_grid_A, start_location, new_angle_A, offset_from_base_A)
    _, contacts_xyz_moved_B, _, _, _, _, _, _, grid_valid_B = implant_grid(gm_mask, orig_grid_B, start_location_B, new_angle_B, offset_from_base_B)

    # 컨택트 병합
    contacts_xyz_moved = np.hstack((contacts_xyz_moved_A, contacts_xyz_moved_B))

    # ROI별 포스핀 파라미터
    phosphenes =    get_phosphenes(contacts_xyz_moved, good_coords,     polar_map, ecc_map, sigma_map)
    phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1,  polar_map, ecc_map, sigma_map)
    phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2,  polar_map, ecc_map, sigma_map)
    phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3,  polar_map, ecc_map, sigma_map)

    # CMF 기반 크기
    M = 1 / get_cortical_magnification(phosphenes_V1[:,1], cort_mag_model)
    spread = cortical_spread(amp)
    sizes = spread * M
    sigmas = sizes / 2.0
    phosphenes_V1[:,2] = sigmas

    # 포스핀 맵 생성
    phospheneMap = np.zeros((WINDOWSIZE, WINDOWSIZE), 'float32')
    phospheneMap = prf_to_phos(phospheneMap, phosphenes_V1, view_angle=view_angle, phSizeScale=1)
    max_val = phospheneMap.max() if phospheneMap.size > 0 else 0
    if max_val > 0:
        phospheneMap /= max_val
    ssum = phospheneMap.sum()
    if ssum > 0:
        phospheneMap /= ssum

    # 타깃 정규화
    target_density = target_density.copy()
    target_density /= target_density.max()
    target_density /= target_density.sum()

    # 손실 계산
    bin_thresh = np.percentile(target_density, dc_percentile)
    dice, _, _ = DC(target_density, phospheneMap, bin_thresh)
    par1 = 1.0 - (a * dice)
    grid_yield = get_yield(contacts_xyz_moved, good_coords)
    par2 = 1.0 - (b * grid_yield)
    hell_d = hellinger_distance(phospheneMap.flatten(), target_density.flatten())
    par3 = c * hell_d if not (np.isnan(hell_d) or np.isinf(hell_d)) else 1.0

    cost = par1 + par2 + par3

    # 충돌/최소거리 제약
    distAB = _min_interprobe_distance_mm(contacts_xyz_moved_A, contacts_xyz_moved_B)
    collide_penalty = 0.0
    if not (grid_valid_A and grid_valid_B):
        cost = par1 + 0.25 + par2 + 0.25 + par3 + 0.25
    # collision penalty temporarily disabled

    if np.isnan(cost) or np.isinf(cost):
        cost = 3.0

    grid_valid = bool(grid_valid_A and grid_valid_B)
    return grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap

###################
## Main Sim Loop ##
###################

def main():
    start_time = time.time()

    # 파일명
    fname_ang = 'inferred_angle.mgz'
    fname_ecc = 'inferred_eccen.mgz'
    fname_sigma = 'inferred_sigma.mgz'
    fname_anat = 'T1.mgz'
    fname_aparc = 'aparc+aseg.mgz'
    fname_label = 'inferred_varea.mgz'

    # 반구별 beta 범위
    dim2_lh_A = Integer(name='beta_A', low=-15, high=110)
    dim2_rh_A = Integer(name='beta_A', low=-110, high=15)
    dim2_lh_B = Integer(name='beta_B', low=-15, high=110)
    dim2_rh_B = Integer(name='beta_B', low=-110, high=15)

    # 대상 피험자(단일)
    subj_list = [100610]
    print('number of subjects: ' + str(len(subj_list)))

    for target_density, ftarget in zip(targ_comb, targ_names):
        for (a, b, c), floss in zip(loss_comb, loss_names):
            # 타깃 정규화 및 이항화 미리보기(저장은 안함)
            target_density = target_density.copy()
            target_density /= target_density.max()
            target_density /= target_density.sum()
            bin_thresh = np.percentile(target_density, dc_percentile)
            target_density_bin = (target_density > bin_thresh).astype(bool)

            for s in subj_list:
                data_dir = datafolder
                if s == 'fsaverage':
                    data_dir = datafolder + str(s) + '/mri/'

                # 데이터 로드
                polar_map = nib.load(data_dir+fname_ang).get_fdata()
                ecc_map   = nib.load(data_dir+fname_ecc).get_fdata()
                sigma_map = nib.load(data_dir+fname_sigma).get_fdata()
                aparc_roi = nib.load(data_dir+fname_aparc).get_fdata()
                label_map = nib.load(data_dir+fname_label).get_fdata()

                # 유효 좌표
                dot = (ecc_map * polar_map)
                good_coords = np.asarray(np.where(dot != 0.0))

                # 반구 필터
                cs_coords_rh = np.where(aparc_roi == 1021)
                cs_coords_lh = np.where(aparc_roi == 2021)
                gm_coords_rh = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
                gm_coords_lh = np.where(aparc_roi > 2000)
                xl,yl,zl = get_xyz(gm_coords_lh)
                xr,yr,zr = get_xyz(gm_coords_rh)
                GM_LH = np.array([xl,yl,zl]).T
                GM_RH = np.array([xr,yr,zr]).T

                # 라벨 분할
                V1_coords_rh = np.asarray(np.where(label_map == 1))
                V1_coords_lh = np.asarray(np.where(label_map == 1))
                V2_coords_rh = np.asarray(np.where(label_map == 2))
                V2_coords_lh = np.asarray(np.where(label_map == 2))
                V3_coords_rh = np.asarray(np.where(label_map == 3))
                V3_coords_lh = np.asarray(np.where(label_map == 3))

                good_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                good_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V1_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V1_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V2_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V2_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V3_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V3_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T

                # 칼카린 고랑 중심
                median_lh = [np.median(cs_coords_lh[0][:]), np.median(cs_coords_lh[1][:]), np.median(cs_coords_lh[2][:])]
                median_rh = [np.median(cs_coords_rh[0][:]), np.median(cs_coords_rh[1][:]), np.median(cs_coords_rh[2][:])]

                gm_mask = np.where(aparc_roi != 0)
                print('target: ', ftarget)
                print('loss: ', floss)
                print('a,b,c: ', a,b,c)

                # 반구별 적용
                for gm_mask_use, hem, start_location, good_c, good_V1, good_V2, good_V3, dim2_A, dim2_B in \
                    zip([GM_LH, GM_RH],
                        ['LH', 'RH'],
                        [median_lh, median_rh],
                        [good_coords_lh, good_coords_rh],
                        [V1_coords_lh, V1_coords_rh],
                        [V2_coords_lh, V2_coords_rh],
                        [V3_coords_lh, V3_coords_rh],
                        [dim2_lh_A, dim2_rh_A],
                        [dim2_lh_B, dim2_rh_B]):

                    # 결과 파일 존재 체크
                    grid_tag = f"{grid_nx}x{grid_ny}x{grid_nz}+{grid2_nx}x{grid2_ny}x{grid2_nz}"
                    data_id = f"{s}_{hem}_V1_n{num_calls}_{grid_tag}_{floss}_{str(dc_percentile)}_{ftarget}"
                    fname = outputfolder + data_id + '.pkl'
                    plot_filename = outputfolder + data_id + '_phosphene_map.png'
                    if os.path.exists(fname):
                        if not os.path.exists(plot_filename):
                            with open(fname, 'rb') as file:
                                stored = pickle.load(file)
                            # stored structure: [res_slim, grid_valid, dice, hell_d, grid_yield,
                            #                    contacts_xyz_moved, good_coords, good_coords_V1,
                            #                    good_coords_V2, good_coords_V3,
                            #                    phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3]
                            stored_phosphenes_V1 = stored[11]
                            phospheneMap = np.zeros((WINDOWSIZE, WINDOWSIZE), 'float32')
                            phospheneMap = prf_to_phos(phospheneMap, stored_phosphenes_V1, view_angle=view_angle, phSizeScale=1)
                            max_val = phospheneMap.max() if phospheneMap.size > 0 else 0
                            if max_val > 0:
                                phospheneMap /= max_val
                            ssum = phospheneMap.sum()
                            if ssum > 0:
                                phospheneMap /= ssum
                            fig, (ax1, ax2) = plt.subplots(1, 2)
                            fig.suptitle('target vs. raw phospheneMap (restored)')
                            ax1.imshow(target_density, cmap='seismic', vmin=0, vmax=np.max(target_density))
                            ax1.set_title('Target map')
                            ax1.axis('off')
                            ax2.imshow(phospheneMap, cmap='seismic', vmin=0, vmax=np.max(phospheneMap) if max_val > 0 else 1)
                            ax2.set_title('Raw phosphene map')
                            ax2.axis('off')
                            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                            plt.close()
                            print('    Restored plot saved as: ', plot_filename)
                        else:
                            print(str(s), ' ', str(hem), ' ',  str(ftarget), ' ', str(floss), ' already processed.')
                        continue

                    # 반구에 맞춘 beta 차원 대체
                    dimA2_use = dim2_A
                    dimB2_use = dim2_B
                    dimensions = [dimA1, dimA2_use, dimA3, dimA4, dimB1, dimB2_use, dimB3, dimB4]

                    @use_named_args(dimensions=dimensions)
                    def f(alpha_A, beta_A, offset_from_base_A, shank_length_A,
                          alpha_B, beta_B, offset_from_base_B, shank_length_B):
                        penalty = 0.25
                        new_angle_A = (float(alpha_A), float(beta_A), 0)
                        new_angle_B = (float(alpha_B), float(beta_B), 0)

                        # Grid A
                        orig_grid_A = create_grid_v2(
                            start_location,
                            shank_length=shank_length_A,
                            n_x=grid_nx, n_y=grid_ny, n_z=grid_nz,
                            spacing_x=spacing_x, spacing_y=spacing_y,
                            offset_from_origin=0
                        )
                        # Grid B (오프셋)
                        start_location_B = [
                            start_location[0] + start_location_B_offset[0],
                            start_location[1] + start_location_B_offset[1],
                            start_location[2] + start_location_B_offset[2],
                        ]
                        orig_grid_B = create_grid_v2(
                            start_location_B,
                            shank_length=shank_length_B,
                            n_x=grid2_nx, n_y=grid2_ny, n_z=grid2_nz,
                            spacing_x=spacing2_x, spacing_y=spacing2_y,
                            offset_from_origin=0
                        )

                        # Implant
                        _, contacts_xyz_moved_A, _, _, _, _, _, _, grid_valid_A = implant_grid(gm_mask_use, orig_grid_A, start_location, new_angle_A, offset_from_base_A)
                        _, contacts_xyz_moved_B, _, _, _, _, _, _, grid_valid_B = implant_grid(gm_mask_use, orig_grid_B, start_location_B, new_angle_B, offset_from_base_B)

                        # 컨택트 합침
                        contacts_xyz_moved = np.hstack((contacts_xyz_moved_A, contacts_xyz_moved_B))

                        # 포스핀 입력
                        phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_V1, polar_map, ecc_map, sigma_map)

                        # CMF 크기
                        M = 1 / get_cortical_magnification(phosphenes_V1[:,1], cort_mag_model)
                        spread = cortical_spread(amp)
                        sizes = spread * M
                        sigmas = sizes / 2.0
                        phosphenes_V1[:,2] = sigmas

                        # 포스핀 맵
                        phospheneMap = np.zeros((WINDOWSIZE, WINDOWSIZE), 'float32')
                        phospheneMap = prf_to_phos(phospheneMap, phosphenes_V1, view_angle=view_angle, phSizeScale=1)
                        max_val = phospheneMap.max() if phospheneMap.size > 0 else 0
                        if max_val > 0:
                            phospheneMap /= max_val
                        ssum = phospheneMap.sum()
                        if ssum > 0:
                            phospheneMap /= ssum

                        # 타깃 사본 정규화
                        tgt = target_density.copy()
                        tgt /= tgt.max()
                        tgt /= tgt.sum()

                        # 손실
                        bin_thresh = np.percentile(tgt, dc_percentile)
                        dice, _, _ = DC(tgt, phospheneMap, bin_thresh)
                        par1 = 1.0 - (a * dice)
                        grid_yield = get_yield(contacts_xyz_moved, good_c)
                        par2 = 1.0 - (b * grid_yield)
                        hell_d = hellinger_distance(phospheneMap.flatten(), tgt.flatten())
                        par3 = 1.0 if (np.isnan(hell_d) or np.isinf(hell_d)) else c * hell_d
                        cost = par1 + par2 + par3

                        # 충돌/최소거리
                        distAB = _min_interprobe_distance_mm(contacts_xyz_moved_A, contacts_xyz_moved_B)
                        collide_penalty = 0.0
                        if not (grid_valid_A and grid_valid_B):
                            cost = par1 + penalty + par2 + penalty + par3 + penalty
                        # collision penalty temporarily disabled

                        if np.isnan(cost) or np.isinf(cost):
                            cost = 3.0

                        print('    ', "{:.2f}".format(cost),
                              "{:.2f}".format(dice),
                              "{:.2f}".format(grid_yield),
                              "{:.2f}".format(par3),
                              (grid_valid_A and grid_valid_B))
                        return cost

                    # 초기 샘플러 및 최적화
                    lhs2 = cook_initial_point_generator("lhs", criterion="maximin")
                    res = gp_minimize(f, x0=x0, dimensions=dimensions, n_jobs=-1,
                                      n_calls=num_calls, n_initial_points=num_initial_points,
                                      initial_point_generator=lhs2, callback=[custom_stopper])

                    print('subject ', s, ' ', hem,
                          ', best (A) alpha,beta,offset,shank: ', res.x[0], res.x[1], res.x[2], res.x[3],
                          ', best (B) alpha,beta,offset,shank: ', res.x[4], res.x[5], res.x[6], res.x[7])

                    # 재현 및 저장용 계산
                    grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap = f_manual(
                        res.x[0], res.x[1], res.x[2], res.x[3],
                        res.x[4], res.x[5], res.x[6], res.x[7],
                        good_c, good_V1, good_V2, good_V3,
                        target_density,
                        start_location, gm_mask_use, polar_map, ecc_map, sigma_map,
                        a, b, c)

                    # 결과 플롯 저장 (왼쪽: 타깃 맵, 오른쪽: raw 포스핀 맵)
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle('target vs. raw phospheneMap')
                    ax1.imshow(target_density, cmap='seismic', vmin=0, vmax=np.max(target_density))
                    ax1.set_title('Target map')
                    ax1.axis('off')
                    ax2.imshow(phospheneMap, cmap='seismic', vmin=0, vmax=np.max(phospheneMap))
                    ax2.set_title('Raw phosphene map')
                    ax2.axis('off')
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    print('    max phospheneMap: ', np.max(phospheneMap))
                    print('    Plot saved as: ', plot_filename)

                    # 결과 객체 저장(슬림)
                    res_slim = {
                        'x': getattr(res, 'x', None),
                        'fun': getattr(res, 'fun', None),
                        'x_iters': getattr(res, 'x_iters', []),
                        'func_vals': getattr(res, 'func_vals', []),
                        'n_calls': getattr(res, 'n_calls', len(getattr(res, 'x_iters', []))),
                        'models_len': len(getattr(res, 'models', []))
                    }
                    with open(fname, 'wb') as file:
                        pickle.dump([res_slim,
                                     grid_valid,
                                     dice, hell_d,
                                     grid_yield,
                                     contacts_xyz_moved,
                                     good_c,
                                     good_V1,
                                     good_V2,
                                     good_V3,
                                     phosphenes,
                                     phosphenes_V1,
                                     phosphenes_V2,
                                     phosphenes_V3], file, protocol=-1)

                    # 텍스트 요약 저장
                    txt_filename = outputfolder + data_id + '.txt'
                    with open(txt_filename, 'w') as file:
                        file.write(f"Subject: {s}\n")
                        file.write(f"Hemisphere: {hem}\n")
                        file.write(f"Target: {ftarget}\n")
                        file.write(f"Loss: {floss}\n")
                        file.write(f"Threshold(dc_percentile): {dc_percentile}\n")
                        file.write(f"Grid tag: {grid_tag}\n")
                        file.write(f"Best parameters:\n")
                        file.write(f"  Probe A - Alpha: {res.x[0]}\n")
                        file.write(f"  Probe A - Beta: {res.x[1]}\n")
                        file.write(f"  Probe A - Offset from base: {res.x[2]}\n")
                        file.write(f"  Probe A - Shank length: {res.x[3]}\n")
                        file.write(f"  Probe B - Alpha: {res.x[4]}\n")
                        file.write(f"  Probe B - Beta: {res.x[5]}\n")
                        file.write(f"  Probe B - Offset from base: {res.x[6]}\n")
                        file.write(f"  Probe B - Shank length: {res.x[7]}\n")
                        file.write(f"Results:\n")
                        file.write(f"  Grid valid: {grid_valid}\n")
                        file.write(f"  Dice coefficient: {dice:.6f}\n")
                        file.write(f"  Hellinger distance: {hell_d:.6f}\n")
                        file.write(f"  Grid yield: {grid_yield:.6f}\n")
                        file.write(f"  Max phospheneMap: {np.max(phospheneMap):.6f}\n")
                        file.write(f"  Number of contact points: {len(contacts_xyz_moved)}\n")
                        file.write(f"  Number of V1 coordinates: {len(good_V1)}\n")
                        file.write(f"  Number of V2 coordinates: {len(good_V2)}\n")
                        file.write(f"  Number of V3 coordinates: {len(good_V3)}\n")

    # 총 소요시간
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print("\n" + "="*60)
    print(f"TOTAL EXECUTION TIME: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total seconds: {total_time:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()


