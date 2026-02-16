# -*- coding: utf-8 -*-
"""
VAE 기반 뇌 임플란트 전극 배치 최적화 V2

기존 베이지안 최적화 대신 VAE(Variational Autoencoder)를 사용하여 최적의 전극 배치 파라미터를 찾는 코드입니다.
베이지안 최적화 코드를 기반으로 하여 VAE 최적화로 변환했습니다.

**최적화 대상 4개 변수:**
1. `alpha`: 시각각도 (-90° ~ 90°)
2. `beta`: 시각각도 (-15° ~ 110°)
3. `offset_from_base`: 기본점으로부터의 오프셋 (0 ~ 40mm)
4. `shank_length`: 샹크 길이 (10 ~ 40mm)
"""

import time
import os.path
import pickle
from copy import deepcopy
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score as MI

# VAE 최적화 모듈 import
try:
    from vae_optimizer_v2 import VAE_Optimizer_V2
except ImportError:
    # 같은 디렉토리에 있다면
    import sys
    sys.path.append('.')
    from vae_optimizer_v2 import VAE_Optimizer_V2

########################
### Custom functions ###
########################
from ninimplant import pol2cart, get_xyz
from lossfunc import DC, KL, get_yield, hellinger_distance
from electphos import create_grid, reposition_grid, implant_grid, get_phosphenes, prf_to_phos, gen_dummy_phos, get_cortical_magnification, cortical_spread
import visualsectors as gvs

# ignore warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.seterr(divide='ignore', invalid='ignore')

def create_cost_function(good_coords, good_coords_V1, good_coords_V2, good_coords_V3, 
                        target_density, gm_mask, start_location, a, b, c):
    """
    VAE 최적화를 위한 비용 함수를 생성합니다.
    클로저를 사용하여 필요한 변수들을 캡처합니다.
    """
    
    def cost_function(alpha, beta, offset_from_base, shank_length):
        """
        전극 배치 비용을 계산하는 함수
        베이지안 최적화의 f 함수와 동일한 로직을 사용합니다.
        """
        penalty = 0.25
        new_angle = (float(alpha), float(beta), 0)
        
        try:
            # 그리드 생성
            orig_grid = create_grid(start_location, shank_length, n_contactpoints_shank, 
                                   spacing_along_xy, offset_from_origin=0)
            
            # 그리드 삽입
            _, contacts_xyz_moved, _, _, _, _, _, _, grid_valid = implant_grid(
                gm_mask, orig_grid, start_location, new_angle, offset_from_base)

            # 포스펜 생성
            phosphenes = get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
            phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1, polar_map, ecc_map, sigma_map)
            phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2, polar_map, ecc_map, sigma_map)
            phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3, polar_map, ecc_map, sigma_map)
            
            # 피질 확대율 계산
            M = 1 / get_cortical_magnification(phosphenes_V1[:,1], cort_mag_model)
            spread = cortical_spread(amp)
            sizes = spread * M
            sigmas = sizes / 2
            
            # 포스펜 크기 설정
            phosphenes_V1[:,2] = sigmas

            # 포스펜 맵 생성
            phospheneMap = np.zeros((WINDOWSIZE, WINDOWSIZE), 'float32')
            phospheneMap = prf_to_phos(phospheneMap, phosphenes_V1, view_angle=view_angle, phSizeScale=1)
            phospheneMap /= phospheneMap.max()
            phospheneMap /= phospheneMap.sum()

            # 이진 임계값 계산
            bin_thresh = np.percentile(target_density, dc_percentile)

            # Dice 계수 계산
            dice, im1, im2 = DC(target_density, phospheneMap, bin_thresh)
            par1 = 1.0 - (a * dice)

            # Yield 계산
            grid_yield = get_yield(contacts_xyz_moved, good_coords)
            par2 = 1.0 - (b * grid_yield)

            # Hellinger 거리 계산
            hell_d = hellinger_distance(phospheneMap.flatten(), target_density.flatten())
            
            # 유효성 검사
            if np.isnan(phospheneMap).any() or np.sum(phospheneMap) == 0:
                par1 = 1
                print('    map is nan or 0')
            
            if np.isnan(hell_d) or np.isinf(hell_d):
                par3 = 1
                print('    Hellinger is nan or inf')
            else:
                par3 = c * hell_d
            
            # 비용 함수 조합
            cost = par1 + par2 + par3

            # 그리드가 유효하지 않을 때 페널티 추가
            if not grid_valid:
                cost = par1 + penalty + par2 + penalty + par3 + penalty
            
            # 비용이 유효하지 않을 때
            if np.isnan(cost) or np.isinf(cost):
                cost = 3
            
            print(f'    {cost:.2f} {dice:.2f} {grid_yield:.2f} {par3:.2f} {grid_valid}')
            return cost
            
        except Exception as e:
            print(f'    Error in cost function: {e}')
            return 10.0  # 에러 시 높은 비용 반환
    
    return cost_function

def f_manual(alpha, beta, offset_from_base, shank_length, good_coords, good_coords_V1, 
             good_coords_V2, good_coords_V3, target_density, start_location, gm_mask):
    """
    최적화된 파라미터로 결과를 시각화하기 위한 함수
    베이지안 최적화의 f_manual 함수와 동일한 로직을 사용합니다.
    """
    
    penalty = 0.25
    new_angle = (float(alpha), float(beta), 0)
    
    # 그리드 생성
    orig_grid = create_grid(start_location, shank_length, n_contactpoints_shank, 
                           spacing_along_xy, offset_from_origin=0)

    # 그리드 삽입
    ref_contacts_xyz, contacts_xyz_moved, refline, refline_moved, projection, ref_orig, ray_visualize, new_location, grid_valid = implant_grid(
        gm_mask, orig_grid, start_location, new_angle, offset_from_base)

    # 포스펜 생성
    phosphenes = get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
    phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1, polar_map, ecc_map, sigma_map)
    phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2, polar_map, ecc_map, sigma_map)
    phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3, polar_map, ecc_map, sigma_map)
    
    # 피질 확대율 계산
    M = 1 / get_cortical_magnification(phosphenes_V1[:,1], cort_mag_model)
    spread = cortical_spread(amp)
    sizes = spread * M
    sigmas = sizes / 2
    
    # 포스펜 크기 설정
    phosphenes_V1[:,2] = sigmas

    # 포스펜 맵 생성
    phospheneMap = np.zeros((WINDOWSIZE, WINDOWSIZE), 'float32')
    phospheneMap = prf_to_phos(phospheneMap, phosphenes_V1, view_angle=view_angle, phSizeScale=1)
    phospheneMap /= phospheneMap.max()
    phospheneMap /= phospheneMap.sum()
    
    # 이진 임계값 계산
    bin_thresh = np.percentile(target_density, dc_percentile)

    # Dice 계수 계산
    dice, im1, im2 = DC(target_density, phospheneMap, bin_thresh)
    par1 = 1.0 - (a * dice)

    # Yield 계산
    grid_yield = get_yield(contacts_xyz_moved, good_coords)
    par2 = 1.0 - (b * grid_yield)
    
    # 타겟 밀도 정규화
    target_density_norm = target_density.copy()
    target_density_norm /= target_density_norm.max()
    target_density_norm /= target_density_norm.sum()
    
    # Hellinger 거리 계산
    hell_d = hellinger_distance(phospheneMap.flatten(), target_density_norm.flatten())
    
    # 유효성 검사
    if np.isnan(phospheneMap).any() or np.sum(phospheneMap) == 0:
        par1 = 1
        print('map is nan or 0')
    
    if np.isnan(hell_d) or np.isinf(hell_d):
        par3 = 1
        print('Hellinger is nan or inf')
    else:
        par3 = c * hell_d
    
    # 비용 함수 조합
    cost = par1 + par2 + par3

    # 그리드가 유효하지 않을 때 페널티 추가
    if not grid_valid:
        cost = par1 + penalty + par2 + penalty + par3 + penalty
    
    # 비용이 유효하지 않을 때
    if np.isnan(cost) or np.isinf(cost):
        cost = 3
    
    return grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap

def main():
    """
    메인 실행 함수
    
    개선된 VAE 최적화를 사용하여 뇌 임플란트 전극 배치를 최적화합니다.
    
    주요 개선 사항:
    1. 안정적인 VAE 학습 (적응적 KL loss, gradient clipping)
    2. 메모리 효율성 (히스토리 크기 제한)
    3. 에러 처리 및 폴백 메커니즘
    4. 베이지안과 유사한 early stopping
    5. 적응적 노이즈 생성
    """
    
    ##########
    ## INIT ##
    ##########
    
    # 데이터 폴더 설정
    datafolder = r'C:\Users\user\YongtaeC\vimplant0812\data\demo\\'
    outputfolder = r'C:\Users\user\YongtaeC\vimplant0812\data\output\\'
    
    # VAE 최적화 파라미터
    param_bounds = [
        (-90, 90),      # alpha: 시각각도
        (-15, 110),     # beta: 시각각도
        (0, 40),        # offset_from_base: mm
        (10, 40)        # shank_length: mm
    ]
    
    # VAE 설정 (개선됨)
    latent_dim = 4          # 잠재 공간 차원 (파라미터 수와 동일하게)
    hidden_dims = [128, 64, 32]  # 더 깊은 네트워크
    device = 'cpu'          # GPU 사용 가능시 'cuda'로 변경
    
    # 최적화 설정 (개선됨)
    num_iterations = 150        # 최적화 반복 횟수 (베이지안과 동일)
    candidates_per_iter = 30    # 반복당 후보 수 (메모리 효율성을 위해 감소)
    initial_samples = 15        # 초기 랜덤 샘플 수 (VAE 학습 안정성을 위해 증가)
    exploration_decay = 0.98    # 탐색 감소율 (더 점진적인 감소)
    
    # 기존 설정들 (베이지안과 동일)
    global dc_percentile, n_contactpoints_shank, spacing_along_xy, WINDOWSIZE
    global cort_mag_model, view_angle, amp, a, b, c
    global polar_map, ecc_map, sigma_map
    
    dc_percentile = 50
    n_contactpoints_shank = 10
    spacing_along_xy = 1
    WINDOWSIZE = 1000
    
    # 손실 함수 가중치 (베이지안과 동일)
    loss_comb = [(1, 0.1, 1)]  # dice, yield, hellinger distance
    loss_names = ['dice-yield-HD']
    
    # 타겟 맵 설정 (베이지안과 동일)
    targ_comb = [
        gvs.upper_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False),
        gvs.lower_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False),
        gvs.inner_ring(windowsize=WINDOWSIZE, fwhm=400, radiusLow=0, radiusHigh=250, plotting=False),
        gvs.complete_gauss(windowsize=1000, fwhm=1200, radiusLow=0, radiusHigh=500, center=None, plotting=False)
    ]
    targ_names = ['targ-upper', 'targ-lower', 'targ-inner', 'targ-full']
    
    # 상수 설정 (베이지안과 동일)
    cort_mag_model = 'wedge-dipole'
    view_angle = 90
    amp = 100
    
    # 피험자 리스트 (베이지안과 동일)
    subj_list = [100206]
    
    ###################
    ## Main Sim Loop ##
    ###################
    
    # 파일명 설정
    fname_ang = 'inferred_angle.mgz'
    fname_ecc = 'inferred_eccen.mgz'
    fname_sigma = 'inferred_sigma.mgz'
    fname_anat = 'T1.mgz'
    fname_aparc = 'aparc+aseg.mgz'
    fname_label = 'inferred_varea.mgz'
    
    print('number of subjects: ' + str(len(subj_list)))
    
    # 반구별 beta 각도 제약 설정 (베이지안과 동일)
    param_bounds_lh = [
        (-90, 90),      # alpha
        (-15, 110),     # beta (LH)
        (0, 40),        # offset_from_base
        (10, 40)        # shank_length
    ]
    
    param_bounds_rh = [
        (-90, 90),      # alpha
        (-110, 15),     # beta (RH)
        (0, 40),        # offset_from_base
        (10, 40)        # shank_length
    ]
    
    # 타겟 맵과 손실 함수 조합을 순회
    for target_density, ftarget in zip(targ_comb, targ_names):
        for (a, b, c), floss in zip(loss_comb, loss_names):
            
            # 타겟 설정
            target_density_norm = target_density.copy()
            target_density_norm /= target_density_norm.max()
            target_density_norm /= target_density_norm.sum()
            
            # 이진 임계값 계산
            bin_thresh = np.percentile(target_density_norm, dc_percentile)
            target_density_bin = (target_density_norm > bin_thresh).astype(bool)
            
            # 타겟 맵 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Target: {ftarget}')
            ax1.imshow(target_density_norm, cmap='seismic')
            ax1.set_title('Target Density')
            ax2.imshow(target_density_bin, cmap='seismic')
            ax2.set_title('Binary Target')
            plt.savefig(outputfolder + f'target_{ftarget}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 각 피험자에 대해 처리
            for s in subj_list:
                print(f'\n=== Processing Subject {s} ===')
                
                data_dir = datafolder
                if s == 'fsaverage':
                    data_dir = datafolder + str(s) + '/mri/'
                
                # 맵 로드
                ang_img = nib.load(data_dir + fname_ang)
                polar_map = ang_img.get_fdata()
                ecc_img = nib.load(data_dir + fname_ecc)
                ecc_map = ecc_img.get_fdata()
                sigma_img = nib.load(data_dir + fname_sigma)
                sigma_map = sigma_img.get_fdata()
                aparc_img = nib.load(data_dir + fname_aparc)
                aparc_roi = aparc_img.get_fdata()
                label_img = nib.load(data_dir + fname_label)
                label_map = label_img.get_fdata()

                # 유효한 복셀 계산
                dot = (ecc_map * polar_map)
                good_coords = np.asarray(np.where(dot != 0.0))

                # 반구별 GM 필터링
                cs_coords_rh = np.where(aparc_roi == 1021)
                cs_coords_lh = np.where(aparc_roi == 2021)
                gm_coords_rh = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
                gm_coords_lh = np.where(aparc_roi > 2000)
                
                xl, yl, zl = get_xyz(gm_coords_lh)
                xr, yr, zr = get_xyz(gm_coords_rh)
                GM_LH = np.array([xl, yl, zl]).T
                GM_RH = np.array([xr, yr, zr]).T

                # 라벨 추출
                V1_coords_rh = np.asarray(np.where(label_map == 1))
                V1_coords_lh = np.asarray(np.where(label_map == 1))
                V2_coords_rh = np.asarray(np.where(label_map == 2))
                V2_coords_lh = np.asarray(np.where(label_map == 2))
                V3_coords_rh = np.asarray(np.where(label_map == 3))
                V3_coords_lh = np.asarray(np.where(label_map == 3))

                # 반구별 V1 좌표 분리
                good_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                good_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V1_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V1_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V2_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V2_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V3_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V3_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T

                # 좌우 반구 중심점 찾기
                median_lh = [np.median(cs_coords_lh[0][:]), np.median(cs_coords_lh[1][:]), np.median(cs_coords_lh[2][:])]
                median_rh = [np.median(cs_coords_rh[0][:]), np.median(cs_coords_rh[1][:]), np.median(cs_coords_rh[2][:])]

                # GM 마스크 계산
                gm_mask = np.where(aparc_roi != 0)
                
                print(f'Target: {ftarget}')
                print(f'Loss: {floss}')
                print(f'a,b,c: {a}, {b}, {c}')

                # 각 반구에 대해 최적화 적용
                for gm_mask, hem, start_location, good_coords, good_coords_V1, good_coords_V2, good_coords_V3, param_bounds in zip(
                    [GM_LH, GM_RH], ['LH', 'RH'], [median_lh, median_rh], 
                    [good_coords_lh, good_coords_rh], [V1_coords_lh, V1_coords_rh], 
                    [V2_coords_lh, V2_coords_rh], [V3_coords_lh, V3_coords_rh], 
                    [param_bounds_lh, param_bounds_rh]):
                    
                    print(f'\n--- Processing {hem} Hemisphere ---')
                    
                    # 이미 처리된 결과 확인
                    data_id = f'{s}_{hem}_V1_VAE_V2_{floss}_{ftarget}'
                    fname = outputfolder + data_id + '.pkl'
                    
                    if os.path.exists(fname):
                        print(f'{s} {hem} {ftarget} {floss} already processed.')
                        continue
                    
                    # VAE 최적화기 초기화
                    vae_opt = VAE_Optimizer_V2(
                        param_bounds=param_bounds,
                        latent_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        device=device
                    )
                    
                    # 비용 함수 생성
                    cost_func = create_cost_function(
                        good_coords, good_coords_V1, good_coords_V2, good_coords_V3,
                        target_density_norm, gm_mask, start_location, a, b, c
                    )
                    
                    # VAE 최적화 실행 (시간 측정 포함)
                    print(f'Starting VAE optimization for {hem} hemisphere...')
                    start_time = time.time()
                    try:
                        best_params, best_cost = vae_opt.optimize(
                            cost_function=cost_func,
                            n_iterations=num_iterations,
                            n_candidates_per_iter=candidates_per_iter,
                            initial_samples=initial_samples,
                            exploration_decay=exploration_decay
                        )
                        optimization_time = time.time() - start_time
                        total_time = optimization_time
                        print(f'VAE optimization completed in {optimization_time:.2f} seconds')
                    except Exception as e:
                        optimization_time = time.time() - start_time
                        print(f"VAE optimization failed after {optimization_time:.2f} seconds: {e}")
                        print("Falling back to random sampling...")
                        # 랜덤 샘플링으로 폴백 (시간 측정 포함)
                        random_start_time = time.time()
                        best_cost = float('inf')
                        best_params = None
                        for _ in range(50):  # 50번의 랜덤 시도
                            random_params = np.array([
                                np.random.uniform(low, high) for low, high in param_bounds
                            ])
                            try:
                                cost = cost_func(*random_params)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_params = random_params.copy()
                            except:
                                continue
                        
                        random_time = time.time() - random_start_time
                        print(f'Random sampling completed in {random_time:.2f} seconds')
                        
                        if best_params is None:
                            print("All optimization attempts failed. Using default parameters.")
                            best_params = np.array([0, 0, 20, 25])  # 기본값
                            best_cost = cost_func(*best_params)
                        
                        # 총 시간 계산
                        total_time = optimization_time + random_time
                        print(f'Total optimization time: {total_time:.2f} seconds')
                    
                    # 결과 출력
                    print(f'\nSubject {s} {hem} Results:')
                    print(f'Best alpha: {best_params[0]:.2f}')
                    print(f'Best beta: {best_params[1]:.2f}')
                    print(f'Best offset_from_base: {best_params[2]:.2f}')
                    print(f'Best shank_length: {best_params[3]:.2f}')
                    print(f'Best cost: {best_cost:.4f}')
                    print(f'Total optimization time: {total_time:.2f} seconds')
                    
                    # 최적 파라미터로 결과 시각화
                    grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap = f_manual(
                        best_params[0], best_params[1], best_params[2], best_params[3],
                        good_coords, good_coords_V1, good_coords_V2, good_coords_V3, target_density_norm,
                        start_location, gm_mask
                    )
                    
                    print(f'Final metrics - Dice: {dice:.4f}, Yield: {grid_yield:.4f}, Hellinger: {hell_d:.4f}')
                    
                    # 결과 포스펜 맵 시각화
                    bin_thresh = np.percentile(phospheneMap, dc_percentile)
                    phospheneMap_bin = (phospheneMap > bin_thresh).astype(bool)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    fig.suptitle(f'{hem} Hemisphere - VAE Optimization Results')
                    ax1.imshow(phospheneMap_bin, cmap='seismic', vmin=0, vmax=np.max(phospheneMap)/100)
                    ax1.set_title('Binary Phosphene Map')
                    ax2.imshow(phospheneMap, cmap='seismic', vmin=0, vmax=np.max(phospheneMap)/100)
                    ax2.set_title('Raw Phosphene Map')
                    plt.savefig(outputfolder + f'{s}_{hem}_{ftarget}_{floss}_results.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f'Max phospheneMap: {np.max(phospheneMap):.6f}')
                    
                    # 최적화 히스토리 시각화
                    try:
                        history_plot_path = outputfolder + f'{s}_{hem}_{ftarget}_{floss}_history.png'
                        vae_opt.plot_optimization_history(save_path=history_plot_path)
                    except Exception as e:
                        print(f"Failed to plot optimization history: {e}")
                    
                    # 결과 저장 (베이지안과 동일한 형식)
                    data_id = f'{s}_{hem}_V1_VAE_V2_{floss}_{ftarget}'
                    fname = outputfolder + data_id + '.pkl'
                    
                    # 베이지안과 동일한 형식으로 저장 (시간 정보 포함)
                    with open(fname, 'wb') as file:
                        pickle.dump([
                            {'x': best_params, 'fun': best_cost, 'time': total_time},  # 베이지안 결과와 동일한 형식 + 시간
                            grid_valid, 
                            dice, hell_d, 
                            grid_yield, 
                            contacts_xyz_moved,
                            good_coords,
                            good_coords_V1,
                            good_coords_V2,
                            good_coords_V3,
                            phosphenes,
                            phosphenes_V1,
                            phosphenes_V2,
                            phosphenes_V3
                        ], file, protocol=-1)
                    
                    print(f'Results saved to {fname}')
                    
                    # VAE 결과도 별도 저장
                    try:
                        vae_results_fname = outputfolder + data_id + '_VAE_model.pkl'
                        vae_opt.save_results(vae_results_fname, optimization_time=total_time)
                        print(f'VAE model saved to {vae_results_fname}')
                        
                        # 최적화 요약 정보 출력
                        summary = vae_opt.get_optimization_summary(optimization_time=total_time)
                        print(f"\nOptimization Summary:")
                        for key, value in summary.items():
                            print(f"  {key}: {value}")
                    except Exception as e:
                        print(f"Failed to save VAE model: {e}")
    
    # 결과 분석 및 비교
    print("\n=== VAE Optimization V2 Completed ===")
    print("All results have been saved to the output folder.")
    print("\nYou can now:")
    print("1. Compare VAE results with Bayesian optimization")
    print("2. Analyze the optimization history")
    print("3. Load and reuse trained VAE models")
    print("4. Fine-tune hyperparameters for better performance")

if __name__ == "__main__":
    main()
