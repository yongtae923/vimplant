# -*- coding: utf-8 -*-
"""
CMA-ES 기반 최적화기 V1 (베이지안 최적화 대체)

베이지안 최적화를 CMA-ES로 교체하여 4차원 파라미터 최적화에 최적화된 버전입니다.
CMA-ES는 4차원에서 매우 효율적이며, 적응적 탐색으로 빠른 수렴을 제공합니다.

@authors: R. van Hoof & A. Lozano (원본 베이지안 코드 기반)
@modified: CMA-ES 최적화로 교체
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

# CMA-ES 라이브러리 import
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    print("Warning: CMA-ES library not available. Please install with: pip install cma")
    CMA_AVAILABLE = False

########################
### Custom functions ###
########################
from ninimplant import pol2cart, get_xyz
from lossfunc import DC, KL, get_yield, hellinger_distance
from electphos import create_grid, reposition_grid, implant_grid, get_phosphenes, prf_to_phos, gen_dummy_phos, get_cortical_magnification, cortical_spread
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

# datafolder = '/path/to/data/'
# outputfolder = '/path/to/data/'
# datafolder = 'F:/Rick/Surfdrive_BACKUP/Data/NESTOR/HCP/subjects/'
# datafolder = r'C:/Users/Lozano/Desktop/NIN/bayesian_optimization_paper/data/subjects/'
# outputfolder =  r'C:/Users/Lozano/Desktop/NIN/bayesian_optimization_paper/data/output/'
datafolder = "C:/Users/user/YongtaeC/vimplant0812/data/input/100610/"
outputfolder = "C:/Users/user/YongtaeC/vimplant0812/data/output/100610_cmaes/"
os.makedirs(outputfolder, exist_ok=True)

# determine range of parameters used in optimization
# CMA-ES는 연속 파라미터를 선호하므로 Real로 변경
dim1 = (-90, 90)      # alpha: visual degrees 
dim2 = (-15, 110)     # beta: visual degrees
dim3 = (0, 40)        # offset_from_base: in mm
dim4 = (10, 40)       # shank_length: mm
dimensions = [dim1, dim2, dim3, dim4]

num_calls = 150
x0 = np.array([0, 0, 20, 25])  # initial values for the four dimensions
num_initial_points = 10
dc_percentile = 50
n_contactpoints_shank = 10
spacing_along_xy = 1
WINDOWSIZE = 1000

# lists of loss term combinations to loop through
    # Dice Coefficient (1, 0, 0)
    # Yield (0, 1, 0)
    # Hellinger Distance (0, 0, 1)
loss_comb = ([(1, 0.1, 1)]) # weights for loss terms
loss_names = (['dice-yield-HD']) # substring in output filename

# lists of target maps to loop through
targ_comb = ([gvs.upper_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False), 
              gvs.lower_sector(windowsize=WINDOWSIZE, fwhm=800, radiusLow=0, radiusHigh=500, plotting=False),
              gvs.inner_ring(windowsize=WINDOWSIZE, fwhm=400, radiusLow=0, radiusHigh=250, plotting=False),
              gvs.complete_gauss(windowsize=1000, fwhm=1200, radiusLow=0, radiusHigh=500, center=None, plotting=False)])
targ_names = (['targ-upper', 'targ-lower', 'targ-inner', 'targ-full'])

# constants pRF model
cort_mag_model = 'wedge-dipole' # which cortex model to use for the cortical magnification
view_angle = 90 #in degrees of visual angle
amp = 100 #stimulation amplitude in micro-amp (higher stimulation -> more tissue activated)

# CMA-ES 설정
sigma0 = 10.0  # 초기 스텝 사이즈
popsize = 10   # 개체 수 (4차원에 적합)

################################################
## Functions related to CMA-ES optimization ##
################################################

def get_evaluation_count(es):
    """CMA-ES 객체에서 평가 횟수를 안전하게 가져오는 함수"""
    try:
        # 최신 버전
        if hasattr(es.result, 'evals_total'):
            return es.result.evals_total
        elif hasattr(es, 'evals_total'):
            return es.evals_total
        elif hasattr(es.result, 'evals'):
            return es.result.evals
        elif hasattr(es, 'evals'):
            return es.evals
        else:
            # 기본값 반환 - generation 속성도 안전하게 확인
            if hasattr(es, 'generation'):
                return es.generation * es.popsize
            elif hasattr(es, 'countiter'):
                return es.countiter * es.popsize
            elif hasattr(es, 'countgen'):
                return es.countgen * es.popsize
            else:
                # 모든 속성이 없으면 기본값
                return 400  # 일반적인 기본값
    except:
        # 오류 발생 시 기본값 반환
        try:
            if hasattr(es, 'popsize'):
                if hasattr(es, 'generation'):
                    return es.generation * es.popsize
                elif hasattr(es, 'countiter'):
                    return es.countiter * es.popsize
                elif hasattr(es, 'countgen'):
                    return es.countgen * es.popsize
                else:
                    return 400
            else:
                return 400
        except:
            return 400

def custom_stopper_cmaes(es, N=5, delta=0.2, thresh=0.05):
    '''
    CMA-ES용 custom stopper
    Returns True (stops the optimization) when 
    the difference between best and worst of the best N are below delta AND the best is below thresh
    
    N = last number of cost values to track
    delta = ratio best and worst
    '''
    if len(es.result.fvals) >= N:
        func_vals = np.sort(es.result.fvals)
        worst = func_vals[N - 1]
        best = func_vals[0]
        
        return (abs((best - worst)/worst) < delta) & (best < thresh)
    else:
        return False

def cmaes_callback(es):
    """CMA-ES 콜백 함수 - 진행 상황 모니터링"""
    try:
        # 안전하게 세대 수 가져오기
        if hasattr(es, 'generation'):
            gen = es.generation
        elif hasattr(es, 'countiter'):
            gen = es.countiter
        elif hasattr(es, 'countgen'):
            gen = es.countgen
        else:
            gen = 0
            
        if gen % 10 == 0:
            print(f"Generation {gen}: Best cost = {es.result.fbest:.6f}")
    except:
        pass
    
    # Custom stopper 체크
    if custom_stopper_cmaes(es):
        print("Custom stopping criteria met!")
        return True
    return False

def cmaes_optimizer(cost_function, x0, param_bounds, n_calls=150, sigma0=10.0, popsize=10):
    """
    CMA-ES 최적화 실행
    
    Args:
        cost_function: 비용 함수
        x0: 초기 파라미터
        param_bounds: 파라미터 경계 [(low1, high1), (low2, high2), ...]
        n_calls: 최대 평가 횟수
        sigma0: 초기 스텝 사이즈
        popsize: 개체 수
    
    Returns:
        최적 파라미터와 비용
    """
    if not CMA_AVAILABLE:
        raise ImportError("CMA-ES library not available")
    
    # CMA-ES 설정
    opts = cma.CMAOptions()
    opts.set({
        'maxiter': n_calls // popsize,  # 세대 수
        'popsize': popsize,             # 개체 수
        'CMA_diagonal': True,           # 대각선 공분산 행렬 (4차원에 효율적)
        'CMA_elitist': True,            # 엘리트 전략
        'tolfun': 1e-6,                # 함수 값 수렴 기준
        'tolx': 1e-6,                  # 파라미터 수렴 기준
        'verbose': -1,                  # 출력 최소화
        'seed': 42                      # 재현성을 위한 시드
    })
    
    # 경계 제약 조건을 위한 wrapper 함수
    def bounded_cost_function(x):
        # 파라미터를 경계 내로 클리핑
        x_clipped = np.clip(x, 
                           [bounds[0] for bounds in param_bounds],
                           [bounds[1] for bounds in param_bounds])
        
        # 비용 함수 호출
        try:
            cost = cost_function(*x_clipped)
            return cost
        except Exception as e:
            print(f"Error in cost function: {e}")
            return 1e6  # 큰 페널티
    
    # CMA-ES 실행
    print(f"Starting CMA-ES optimization with {n_calls} max evaluations...")
    print(f"Population size: {popsize}, Initial sigma: {sigma0}")
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    # 최적화 실행
    start_time = time.time()
    es.optimize(bounded_cost_function, callback=cmaes_callback)
    optimization_time = time.time() - start_time
    
    # 평가 횟수 안전하게 가져오기
    eval_count = get_evaluation_count(es)
    
    print(f"\nCMA-ES optimization completed in {optimization_time:.2f} seconds")
    print(f"Best parameters: {es.result.xbest}")
    print(f"Best cost: {es.result.fbest:.6f}")
    print(f"Total evaluations: {eval_count}")
    
    return es.result.xbest, es.result.fbest, es, optimization_time

################################################
## Main optimization function ##
################################################

################################################
## Main optimization loop ##
################################################

def main_optimization():
    """메인 최적화 루프 - CMA-ES 사용 (8개 타겟: 4개 × 2개 반구)"""
    global start_location, gm_mask, good_coords, good_coords_V1, good_coords_V2, good_coords_V3, target_density
    global polar_map, ecc_map, sigma_map
    
    # 전체 실행 시간 측정 시작
    total_start_time = time.time()
    
    print("Starting CMA-ES optimization for 8 targets (4 targets × 2 hemispheres)...")
    print(f"Target combinations: {len(targ_comb)}")
    print(f"Loss combinations: {len(loss_comb)}")
    
    # Load subject data
    subject = '100610'
    
    # Load brain data
    print(f"Loading brain data for subject {subject}...")
    
    # Load T1 and brain mask
    t1_file = os.path.join(datafolder, 'T1.mgz')
    aseg_file = os.path.join(datafolder, 'aparc+aseg.mgz')
    
    t1 = nib.load(t1_file)
    aseg = nib.load(aseg_file)
    
    # Load pRF maps
    polar_file = os.path.join(datafolder, 'inferred_angle.mgz')
    ecc_file = os.path.join(datafolder, 'inferred_eccen.mgz')
    sigma_file = os.path.join(datafolder, 'inferred_sigma.mgz')
    
    polar_map = nib.load(polar_file).get_fdata()
    ecc_map = nib.load(ecc_file).get_fdata()
    sigma_map = nib.load(sigma_file).get_fdata()
    
    # Load V1, V2, V3 area labels
    varea_file = os.path.join(datafolder, 'inferred_varea.mgz')
    varea_map = nib.load(varea_file).get_fdata()
    
    # Create brain mask
    gm_mask = (aseg.get_fdata() == 3) | (aseg.get_fdata() == 42)
    
    # Compute valid voxels (same as bayesianopt_V1.ipynb)
    print("Computing valid coordinates...")
    dot = (ecc_map * polar_map)
    good_coords = np.asarray(np.where(dot != 0.0))
    
    # Filter GM per hemisphere
    cs_coords_rh = np.where(aseg.get_fdata() == 1021)
    cs_coords_lh = np.where(aseg.get_fdata() == 2021)
    gm_coords_rh = np.where((aseg.get_fdata() >= 1000) & (aseg.get_fdata() < 2000))
    gm_coords_lh = np.where(aseg.get_fdata() > 2000)
    
    # Extract V1, V2, V3 coordinates
    V1_coords_rh = np.asarray(np.where(varea_map == 1))
    V1_coords_lh = np.asarray(np.where(varea_map == 1))
    V2_coords_rh = np.asarray(np.where(varea_map == 2))
    V2_coords_lh = np.asarray(np.where(varea_map == 2))
    V3_coords_rh = np.asarray(np.where(varea_map == 3))
    V3_coords_lh = np.asarray(np.where(varea_map == 3))
    
    # Divide coordinates per hemisphere (same logic as bayesianopt_V1.ipynb)
    good_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
    good_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
    V1_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
    V1_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
    V2_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
    V2_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
    V3_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
    V3_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
    
    print(f"Computed coordinates:")
    print(f"  - good_coords_LH: {good_coords_lh.shape}")
    print(f"  - good_coords_RH: {good_coords_rh.shape}")
    print(f"  - V1_coords_LH: {V1_coords_lh.shape}, V1_coords_RH: {V1_coords_rh.shape}")
    print(f"  - V2_coords_LH: {V2_coords_lh.shape}, V2_coords_RH: {V2_coords_rh.shape}")
    print(f"  - V3_coords_LH: {V3_coords_lh.shape}, V3_coords_RH: {V3_coords_rh.shape}")
    
    # Find center of left and right calcarine sulci
    median_lh = [np.median(cs_coords_lh[0][:]), np.median(cs_coords_lh[1][:]), np.median(cs_coords_lh[2][:])]
    median_rh = [np.median(cs_coords_rh[0][:]), np.median(cs_coords_rh[1][:]), np.median(cs_coords_rh[2][:])]
    
    # Convert to numpy arrays for proper 3D coordinate format
    median_lh = np.array(median_lh, dtype=np.float64)
    median_rh = np.array(median_rh, dtype=np.float64)
    
    print(f"Start locations:")
    print(f"  - LH (Left Hemisphere): {median_lh}")
    print(f"  - RH (Right Hemisphere): {median_rh}")
    
    # Define the cost function f inside main_optimization
    def f(alpha, beta, offset_from_base, shank_length):
        """
        This function encapsulates the electrode placement procedure and returns the cost value by 
        comparing the resulting phosphene map with the target map.    
        * First it creats a grid based on the four parameters. 
        * Phosphenes are generated based on the grid's contact points, 
          and their sizes are determined using cortical magnification and spread values. 
        * These phosphenes are converted into a 2D image representation. 
          The function then computes the dice coefficient and yield, and calculates the Hellinger 
          distance between the generated image and a target density. 
        * The resulting cost is a combination of these factors, 
          with penalties applied if the grid is invalid. 
        * The function also handles cases of invalid values and prints diagnostic information. 
        * Ultimately, the function returns the calculated cost used by the CMA-ES algorithm.
        """
        
        penalty = 0.25
        new_angle = (float(alpha), float(beta), 0)    
        
        try:
            # create grid - ensure start_location is proper 3D coordinate
            if start_location is None or len(start_location) != 3:
                print(f"    Error: Invalid start_location: {start_location}")
                return 10.0
            
            # Ensure start_location is a numpy array with proper shape
            start_loc = np.array(start_location, dtype=np.float64).flatten()
            if start_loc.size != 3:
                print(f"    Error: start_location must be 3D, got shape: {start_loc.shape}")
                return 10.0
            
            orig_grid = create_grid(start_loc, shank_length, n_contactpoints_shank, spacing_along_xy, offset_from_origin=0)
            
            # implanting grid
            _, contacts_xyz_moved, _, _, _, _, _, _, grid_valid = implant_grid(gm_mask, orig_grid, start_loc, new_angle, offset_from_base)

            # get angle, ecc and rfsize for contactpoints (phosphenes[0-2][:] 0 angle x 1 ecc x 2 rfsize)    
            phosphenes = get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
            phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1, polar_map, ecc_map, sigma_map)
            phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2, polar_map, ecc_map, sigma_map)   
            phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3, polar_map, ecc_map, sigma_map)
            
            # Check if phosphenes are valid
            if phosphenes_V1.size == 0:
                print('    Error: No valid phosphenes generated')
                return 10.0
            
            #the inverse cortical magnification in degrees (visual angle)/mm tissue
            M = 1 / get_cortical_magnification(phosphenes_V1[:,1], cort_mag_model) 
            spread = cortical_spread(amp) #radius of current spread in the tissue, in mm
            sizes = spread*M #radius of current spread * cortical magnification = rf radius in degrees
            sigmas = sizes / 2  # radius to sigma of gaussian
            
            # phosphene size based on CMF + stim amp
            phosphenes_V1[:,2] = sigmas

            # generate map using Gaussians
            # transforming obtained phosphenes to a 2d image    
            phospheneMap = np.zeros((WINDOWSIZE,WINDOWSIZE), 'float32') 
            phospheneMap = prf_to_phos(phospheneMap, phosphenes_V1, view_angle=view_angle, phSizeScale=1)    
            phospheneMap /= phospheneMap.max()
            phospheneMap /= phospheneMap.sum()

            # bin_thesh determines size target
            bin_thresh=np.percentile(target_density, dc_percentile)# np.min(target_density) 

            # compute dice coefficient -> should be 1 -> invert cost 
            dice, im1, im2 = DC(target_density, phospheneMap, bin_thresh)
            par1 = 1.0 - (amax * dice)

            # compute yield -> should be 1 -> invert cost
            grid_yield = get_yield(contacts_xyz_moved, good_coords)
            par2 = 1.0 - (bmax * grid_yield)

            # compute hellinger distance -> should be small -> keep cost
            hell_d = hellinger_distance(phospheneMap.flatten(), target_density.flatten())        
            
            ## validations steps
            if np.isnan(phospheneMap).any() or np.sum(phospheneMap) == 0:
                par1 = 1
                print('    map is nan or 0')
            
            if np.isnan(hell_d) or np.isinf(hell_d):
                par3 = 1
                print('    Hellinger is nan or inf')
            else:
                par3 = cmax * hell_d
            
            # combine cost functions
            cost = par1 + par2 + par3

            # when some contact points are outside of the hemisphere (convex), add penalty
            if not grid_valid:
                cost = par1 + penalty + par2 + penalty + par3 + penalty
            
            # check if cost contains invalid value
            if np.isnan(cost) or np.isinf(cost):
                cost = 3
            
            print('    ', "{:.2f}".format(cost), "{:.2f}".format(dice), "{:.2f}".format(grid_yield), "{:.2f}".format(par3), grid_valid)
            return cost
            
        except Exception as e:
            print(f'    Error in cost function: {e}')
            return 10.0  # 에러 시 높은 비용 반환
    
    # Define f_manual function inside main_optimization as well
    def f_manual(alpha, beta, offset_from_base, shank_length, good_coords, good_coords_V1, good_coords_V2, good_coords_V3, target_density):
        '''
        Copy from f, to obtain phosphene map and contact points for the optimized parameters. Used to visualize results.
        also returns coords used ect.
        '''
        
        penalty = 0.25
        new_angle = (float(alpha), float(beta), 0)
        
        try:
            # Ensure start_location is proper 3D coordinate
            start_loc = np.array(start_location, dtype=np.float64).flatten()
            if start_loc.size != 3:
                print(f"Error: Invalid start_location in f_manual: {start_location}")
                return False, 0, 0, 0, None, None, None, None, None, None
            
            # create grid
            orig_grid = create_grid(start_loc, shank_length, n_contactpoints_shank, spacing_along_xy, offset_from_origin=0)

            # implanting grid
            ref_contacts_xyz, contacts_xyz_moved, refline, refline_moved, projection, ref_orig, ray_visualize, new_location, grid_valid = implant_grid(gm_mask, orig_grid, start_loc, new_angle, offset_from_base)

            # get angle, ecc and rfsize for contactpoints in each ROI (phosphenes[0-2][:] 0 angle x 1 ecc x 2 rfsize)
            phosphenes =    get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
            phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1, polar_map, ecc_map, sigma_map)
            phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2, polar_map, ecc_map, sigma_map)
            phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3, polar_map, ecc_map, sigma_map)
            
            # Check if phosphenes are valid
            if phosphenes_V1.size == 0:
                print('Error: No valid phosphenes generated in f_manual')
                return False, 0, 0, 0, None, None, None, None, None, None
            
            #the inverse cortical magnification in degrees (visual angle)/mm tissue
            M = 1 / get_cortical_magnification(phosphenes_V1[:,1], cort_mag_model)
            spread = cortical_spread(amp) #radius of current spread in the tissue, in mm
            sizes = spread*M #radius of current spread * cortical magnification = rf radius in degrees
            sigmas = sizes / 2  # radius to sigma of gaussian
            
            # phosphene size based on CMF + stim amp
            phosphenes_V1[:,2] = sigmas

            # generate map using Gaussians
            # transforming obtained phosphenes to a 2d image
            phospheneMap = np.zeros((WINDOWSIZE,WINDOWSIZE), 'float32')
            phospheneMap = prf_to_phos(phospheneMap, phosphenes_V1, view_angle=view_angle, phSizeScale=1)
            phospheneMap /= phospheneMap.max()
            phospheneMap /= phospheneMap.sum()
            
            # can we relate bin_thesh to an eccentricity value? -> taken care of by masking the targets -> CHANGE TO 0.99999
            bin_thresh=np.percentile(target_density, dc_percentile)

            # compute dice coefficient -> should be large -> invert cost 
            dice, im1, im2 = DC(target_density, phospheneMap, bin_thresh)
            par1 = 1.0 - (amax * dice)

            # compute yield -> should be 1 -> invert cost
            grid_yield = get_yield(contacts_xyz_moved, good_coords)
            par2 = 1.0 - (bmax * grid_yield)  
            
            # very important to normalize target density to same range as phospheneMap!
            target_density_norm = target_density.copy()
            target_density_norm /= target_density_norm.max()
            target_density_norm /= target_density_norm.sum()
            
            # compute Hellinger distance -> should be small -> keep cost
            hell_d = hellinger_distance(phospheneMap.flatten(), target_density_norm.flatten())
            
            ## validations steps
            if np.isnan(phospheneMap).any() or np.sum(phospheneMap) == 0:
                par1 = 1
                print('map is nan or 0')
            
            if np.isnan(hell_d) or np.isinf(hell_d):
                par3 = 1
                print('Hellinger is nan or inf')
            else:
                par3 = cmax * hell_d
            
            # combine cost functions
            cost = par1 + par2 + par3

            # when some contact points are outside of the hemisphere (convex), add penalty
            if not grid_valid:
                cost = par1 + penalty + par2 + penalty + par3 + penalty
            
            # check if cost contains invalid value
            if np.isnan(cost) or np.isinf(cost):
                cost = 3
            
            return grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap
            
        except Exception as e:
            print(f'Error in f_manual: {e}')
            return False, 0, 0, 0, None, None, None, None, None, None
    
    # 최적화 결과 저장용 리스트
    optimization_results = []
    
    # Main optimization loop - 8 targets (4 targets × 2 hemispheres)
    for targ_idx, target_density in enumerate(targ_comb):
        for loss_idx, (a, b, c) in enumerate(loss_comb):
            print(f"\n{'='*60}")
            print(f"🎯 TARGET: {targ_names[targ_idx]}")
            print(f"⚖️  LOSS: {loss_names[loss_idx]} (a={a}, b={b}, c={c})")
            print(f"{'='*60}")
            
            # Update global variables for loss function
            global amax, bmax, cmax
            amax, bmax, cmax = a, b, c
            
            # Apply optimization to each hemisphere (like bayesianopt_V1.ipynb)
            for gm_mask, hem, start_location, good_coords, good_coords_V1, good_coords_V2, good_coords_V3 in zip(
                [gm_coords_lh, gm_coords_rh], 
                ['LH', 'RH'], 
                [median_lh, median_rh], 
                [good_coords_lh, good_coords_rh], 
                [V1_coords_lh, V1_coords_rh], 
                [V2_coords_lh, V2_coords_rh], 
                [V3_coords_lh, V3_coords_rh]
            ):
                print(f"\n🔄 Processing {hem} hemisphere...")
                
                # Check if already processed
                data_id = f"{subject}_{hem}_CMAES_{targ_names[targ_idx]}_{loss_names[loss_idx]}"
                fname = os.path.join(outputfolder, data_id + '.pkl')
                
                if os.path.exists(fname):
                    print(f"✅ {subject} {hem} {targ_names[targ_idx]} {loss_names[loss_idx]} already processed.")
                    continue
                
                # Run CMA-ES optimization
                try:
                    print(f"🚀 Starting CMA-ES optimization for {hem}...")
                    best_params, best_cost, es_result, opt_time = cmaes_optimizer(
                        cost_function=f,
                        x0=x0,
                        param_bounds=dimensions,
                        n_calls=num_calls,
                        sigma0=sigma0,
                        popsize=popsize
                    )
                    
                    print(f"\n🎉 Optimization completed successfully for {hem}!")
                    print(f"📊 Best parameters: alpha={best_params[0]:.2f}, beta={best_params[1]:.2f}, offset={best_params[2]:.2f}, shank_length={best_params[3]:.2f}")
                    print(f"💰 Best cost: {best_cost:.6f}")
                    print(f"⏱️  Optimization time: {opt_time:.2f} seconds")
                    
                    # Get detailed results using f_manual (like bayesianopt_V1.ipynb)
                    print(f"🔍 Computing detailed results...")
                    grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap = f_manual(
                        best_params[0], best_params[1], best_params[2], best_params[3], 
                        good_coords, good_coords_V1, good_coords_V2, good_coords_V3, target_density
                    )
                    
                    print(f"📈 Detailed Results for {hem}:")
                    print(f"    🎯 Dice Coefficient: {dice:.6f}")
                    print(f"    📊 Grid Yield: {grid_yield:.6f}")
                    print(f"    📏 Hellinger Distance: {hell_d:.6f}")
                    print(f"    ✅ Grid Valid: {grid_valid}")
                    print(f"    🧠 Contact Points: {contacts_xyz_moved.shape}")
                    
                    # 결과 저장
                    result_info = {
                        'subject': subject,
                        'hemisphere': hem,
                        'target_name': targ_names[targ_idx],
                        'loss_name': loss_names[loss_idx],
                        'best_params': best_params,
                        'best_cost': best_cost,
                        'optimization_time': opt_time,
                        'dice': dice,
                        'grid_yield': grid_yield,
                        'hellinger_distance': hell_d,
                        'grid_valid': grid_valid,
                        'cmaes_result': es_result
                    }
                    optimization_results.append(result_info)
                    
                    # Save results (same format as bayesianopt_V1.ipynb)
                    results = {
                        'best_params': best_params,
                        'best_cost': best_cost,
                        'optimization_time': opt_time,
                        'target_name': targ_names[targ_idx],
                        'loss_name': loss_names[loss_idx],
                        'dice': dice,
                        'grid_yield': grid_yield,
                        'hellinger_distance': hell_d,
                        'grid_valid': grid_valid,
                        'contacts_xyz_moved': contacts_xyz_moved,
                        'good_coords': good_coords,
                        'good_coords_V1': good_coords_V1,
                        'good_coords_V2': good_coords_V2,
                        'good_coords_V3': good_coords_V3,
                        'phosphenes': phosphenes,
                        'phosphenes_V1': phosphenes_V1,
                        'phosphenes_V2': phosphenes_V2,
                        'phosphenes_V3': phosphenes_V3,
                        'cmaes_result': es_result
                    }
                    
                    with open(fname, 'wb') as f:
                        pickle.dump(results, f)
                    
                    print(f"💾 Results saved to: {fname}")
                    
                except Exception as e:
                    print(f"❌ Error in optimization for {hem}: {e}")
                    continue
    
    # 전체 실행 시간 계산
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # 최적화 요약 출력
    print("\n" + "="*80)
    print("🎯 CMA-ES OPTIMIZATION SUMMARY (8 TARGETS)")
    print("="*80)
    
    print(f"📊 Total Execution Time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
    print(f"🎯 Total Targets: {len(targ_comb)}")
    print(f"⚖️  Total Loss Functions: {len(loss_comb)}")
    print(f"🔄 Total Optimizations: {len(optimization_results)}")
    
    if optimization_results:
        print(f"\n📈 Individual Optimization Results:")
        total_opt_time = 0
        best_overall_cost = float('inf')
        best_overall_params = None
        best_overall_dice = 0
        best_overall_yield = 0
        
        for i, result in enumerate(optimization_results):
            print(f"\n  {i+1}. {result['subject']} {result['hemisphere']} - {result['target_name']} - {result['loss_name']}")
            print(f"     🎯 Best Cost: {result['best_cost']:.6f}")
            print(f"     ⏱️  Time: {result['optimization_time']:.2f}s")
            print(f"     📊 Parameters: alpha={result['best_params'][0]:.2f}, beta={result['best_params'][1]:.2f}, offset={result['best_params'][2]:.2f}, shank_length={result['best_params'][3]:.2f}")
            print(f"     🎯 Dice: {result['dice']:.6f}, Yield: {result['grid_yield']:.6f}, HD: {result['hellinger_distance']:.6f}")
            print(f"     ✅ Grid Valid: {result['grid_valid']}")
            
            total_opt_time += result['optimization_time']
            
            if result['best_cost'] < best_overall_cost:
                best_overall_cost = result['best_cost']
                best_overall_params = result['best_params']
                best_overall_dice = result['dice']
                best_overall_yield = result['grid_yield']
        
        print(f"\n🏆 Best Overall Result:")
        print(f"    💰 Cost: {best_overall_cost:.6f}")
        print(f"    📊 Parameters: alpha={best_overall_params[0]:.2f}, beta={best_overall_params[1]:.2f}, offset={best_overall_params[2]:.2f}, shank_length={best_overall_params[3]:.2f}")
        print(f"    🎯 Dice: {best_overall_dice:.6f}, Yield: {best_overall_yield:.6f}")
        print(f"    ⏱️  Total Optimization Time: {total_opt_time:.2f}s")
        print(f"    📁 Data Loading & Setup Time: {total_execution_time - total_opt_time:.2f}s")
    
    print(f"\n✅ All 8 target optimizations completed in {total_execution_time:.2f} seconds!")
    print("="*80)

if __name__ == "__main__":
    main_optimization()
