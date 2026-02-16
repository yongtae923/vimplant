# -*- coding: utf-8 -*-
"""
Created on Wed May 6 2021

@authors: R. van Hoof & A. Lozano

Jupyter 노트북을 Python 스크립트로 변환
"""

import time
import os.path
import pickle # needed to store the results
from copy import deepcopy
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score as MI
from skopt.utils import use_named_args
from skopt.space import Integer, Categorical, Real
from skopt.utils import cook_initial_point_generator
from skopt import gp_minimize

########################
### Custom functions ###
########################
from ninimplant import pol2cart, get_xyz # matrix rotation/translation ect
from lossfunc import DC, KL, get_yield, hellinger_distance
from electphos import create_grid, reposition_grid, implant_grid, get_phosphenes, prf_to_phos, gen_dummy_phos, get_cortical_magnification, cortical_spread, create_grid_v2
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
datafolder = "C:/Users/user/YongtaeC/vimplant0812/data/input/102311/"
outputfolder = "C:/Users/user/YongtaeC/vimplant0812/data/output/102311_py/"
os.makedirs(outputfolder, exist_ok=True)

# determine range of parameters used in optimization
dim1 = Integer(name='alpha', low=-90, high=90) # visual degrees 
dim2 = Integer(name='beta', low=-15, high=110) # visual degrees - -15 is probably lowest possible angle, otherwise other hem is in the way if hem = RH -> low = -110, high = 15
dim3 = Integer(name='offset_from_base', low=0, high=40) # in mm
dim4 = Integer(name='shank_length', low=10, high=40) # mm
dimensions = [dim1, dim2, dim3, dim4]

num_calls = 150
x0 = (0,0,20,25) # initial values for the four dimensions
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
              gvs.complete_gauss(windowsize=1000, fwhm=1200, radiusLow=0, radiusHigh=500, center=None, plotting=False),
              gvs.square_target(windowsize=WINDOWSIZE, fwhm=400, square_size=300, center=None, plotting=False),
              gvs.triangle_target(windowsize=WINDOWSIZE, fwhm=400, triangle_size=300, center=None, plotting=False)])
targ_names = (['targ-upper', 'targ-lower', 'targ-inner', 'targ-full', 'targ-square', 'targ-triangle'])

# constants pRF model
cort_mag_model = 'wedge-dipole' # which cortex model to use for the cortical magnification
view_angle = 90 #in degrees of visual angle
amp = 100 #stimulation amplitude in micro-amp (higher stimulation -> more tissue activated)

# INIT Bayes
amax = 1
bmax = 1
cmax = 1000
N=5
delta=0.2
thresh=0.05

# subjects to include
subj_list = [118225, 144226, 162935, 176542, 187345, 200614, 251833, 389357, 547046, 671855, 789373, 901139,  
100610, 125525, 145834, 164131, 177140, 191033, 201515, 257845, 393247, 550439, 680957, 814649, 901442, 
102311, 126426, 146129, 164636, 177645, 191336, 203418, 263436, 395756, 552241, 690152, 818859, 905147, 
102816, 128935, 146432, 165436, 177746, 191841, 204521, 283543, 397760, 562345, 706040, 825048, 910241, 
104416, 130114, 146735, 167036, 178142, 192439, 205220, 318637, 401422, 572045, 724446, 826353, 926862, 
105923, 130518, 146937, 167440, 178243, 192641, 209228, 320826, 406836, 573249, 725751, 833249, 927359, 
108323, 131217, 148133, 169040, 178647, 193845, 212419, 330324, 412528, 581450, 732243, 859671, 942658, 
109123, 131722, 150423, 169343, 180533, 195041, 214019, 346137, 429040, 585256, 751550, 861456, 943862, 
111312, 132118, 155938, 169444, 181232, 196144, 214524, 352738, 436845, 601127, 757764, 871762, 951457, 
111514, 134627, 156334, 169747, 181636, 197348, 221319, 360030, 463040, 617748, 765864, 872764, 958976, 
114823, 134829, 157336, 171633, 182436, 198653, 233326, 365343, 467351, 627549, 770352, 878776, 966975, 
115017, 135124, 158035, 172130, 182739, 199655, 239136, 380036, 525541, 638049, 771354, 878877, 971160, 
115825, 137128, 158136, 173334, 185442, 200210, 246133, 381038, 536647, 644246, 782561, 898176, 973770, 
116726, 140117, 159239, 175237, 186949, 200311, 249947, 385046, 541943, 654552, 783462, 899885, 995174, 'fsaverage']

subj_list = [100206]
subj_list = [100610]

# 원하는 그리드 크기/간격(예시: 7x7x5)
grid_nx = 10
grid_ny = 10
grid_nz = 10

# x,y 간격(mm). 예전처럼 하나만 쓰고 싶으면 둘 다 같은 값으로.
spacing_x = 0.2
spacing_y = 0.2

# 기존 파라미터는 그대로 사용 가능
shank_length = 25  # (최적화 변수인 경우 그대로 두고, 아니면 고정값)
n_contactpoints_shank = grid_nz      # 하위호환용으로 유지하고 싶으면 맞춰두기(선택)
spacing_along_xy = spacing_x         # 기존 변수 쓰는 곳 있으면 맞춰두기

################################################
## Functions related to Bayesian optimization ##
################################################

def custom_stopper(res, N=5, delta=0.2, thresh=0.05):
    '''
    Returns True (stops the optimization) when 
    the difference between best and worst of the best N are below delta AND the best is below thresh
    
    N = last number of cost values to track
    delta = ratio best and worst
    
    '''
    
    if len(res.func_vals) >= N:
        func_vals = np.sort(res.func_vals)
        worst = func_vals[N - 1]
        best = func_vals[0]
        
        return (abs((best - worst)/worst) < delta) & (best < thresh)
    else:
        return None

# f 함수는 main 함수 내에서 정의됩니다 (클로저로 변수들을 캡처하기 위해)

def f_manual(alpha, beta, offset_from_base, shank_length,
             good_coords, good_coords_V1, good_coords_V2, good_coords_V3,
             target_density,
             start_location, gm_mask, polar_map, ecc_map, sigma_map,
             a, b, c):
    '''
    Copy from f, to obtain phosphene map and contact points for the optimized parameters. Used to visualize results.

    also returns coords used ect.
    '''
    
    penalty = 0.25
    new_angle = (float(alpha), float(beta), 0)
    
    # create grid
    orig_grid = create_grid_v2(
        start_location,
        shank_length=shank_length,
        n_x=grid_nx, n_y=grid_ny, n_z=grid_nz,
        spacing_x=spacing_x, spacing_y=spacing_y,
        offset_from_origin=0
    )

    # implanting grid
    ref_contacts_xyz, contacts_xyz_moved, refline, refline_moved, projection, ref_orig, ray_visualize, new_location, grid_valid = implant_grid(gm_mask, orig_grid, start_location, new_angle, offset_from_base)

    # get angle, ecc and rfsize for contactpoints in each ROI (phosphenes[0-2][:] 0 angle x 1 ecc x 2 rfsize)
    phosphenes =    get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
    phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1, polar_map, ecc_map, sigma_map)
    phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2, polar_map, ecc_map, sigma_map)
    phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3, polar_map, ecc_map, sigma_map)
    
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
    print(view_angle)
    
    # can we relate bin_thesh to an eccentricity value? -> taken care of by masking the targets -> CHANGE TO 0.99999
    bin_thresh=np.percentile(target_density, dc_percentile)# np.min(target_density) # bin_thesh determines size target

    # compute dice coefficient -> should be large -> invert cost 
    dice, im1, im2 = DC(target_density, phospheneMap, bin_thresh)
    par1 = 1.0 - (a * dice)

    # compute yield -> should be 1 -> invert cost
    grid_yield = get_yield(contacts_xyz_moved, good_coords)
    par2 = 1.0 - (b * grid_yield)  
    
    # very important to normalize target density to same range as phospheneMap!
    target_density /= target_density.max()
    target_density /= target_density.sum()        
    
    # compute Hellinger distance -> should be small -> keep cost
    hell_d = hellinger_distance(phospheneMap.flatten(), target_density.flatten())
    
    ## validations steps
    if np.isnan(phospheneMap).any() or np.sum(phospheneMap) == 0:
        par1 = 1
        print('map is nan or 0')
    
    if np.isnan(hell_d) or np.isinf(hell_d):
        par3 = 1
        print('Hellington is nan or inf')
    else:
        par3 = c * hell_d
    
    # combine cost functions
    cost = par1 + par2 + par3

    # when some contact points are outside of the hemisphere (convex), add penalty
    if not grid_valid:
        cost = par1 + penalty + par2 + penalty + par3 + penalty
    
    # check if cost contains invalid value
    if np.isnan(cost) or np.isinf(cost):
        cost = 3
    
    return grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap

###################
## Main Sim Loop ##
###################

def main(output_folder=None):
    # Start timing
    start_time = time.time()
    
    # 출력 폴더 설정 (매개변수로 받거나 기본값 사용)
    if output_folder is None:
        output_folder = "C:/Users/user/YongtaeC/vimplant0812/data/output/102311_py/"
    os.makedirs(output_folder, exist_ok=True)
    
    # set file names
    fname_ang = 'inferred_angle.mgz'
    fname_ecc = 'inferred_eccen.mgz'
    fname_sigma = 'inferred_sigma.mgz'
    fname_anat = 'T1.mgz'
    fname_aparc = 'aparc+aseg.mgz'
    fname_label = 'inferred_varea.mgz'
    print('number of subjects: ' + str(len(subj_list)))

    # set beta angle constraints according to hemisphere
    dim2_lh = Integer(name='beta', low=-15, high=110)
    dim2_rh = Integer(name='beta', low=-110, high=15)

    # loop through phosphene target maps and combinations of loss terms
    for target_density, ftarget in zip(targ_comb, targ_names):
        for (a, b, c), floss in zip(loss_comb, loss_names):
            # set target
            target_density /= target_density.max()
            target_density /= target_density.sum()
            # can we relate bin_thesh to an eccentricity value?
            bin_thresh=np.percentile(target_density, dc_percentile ) #np.min(target_density) # bin_thesh determines size target
            target_density_bin = (target_density > bin_thresh).astype(bool)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Horizontally stacked subplots')
            plt.subplot(1,2,1).imshow(target_density, cmap = 'seismic')
            plt.subplot(1,2,2).imshow(target_density_bin, cmap = 'seismic')

            for s in subj_list:
                # data_dir = datafolder + str(s)+ '/T1w/' + str(s) + '/mri/'            
                data_dir = datafolder

                if s == 'fsaverage':
                    data_dir = datafolder + str(s) + '/mri/'
                    
                # load maps
                ang_img = nib.load(data_dir+fname_ang)
                polar_map = ang_img.get_fdata()
                ecc_img = nib.load(data_dir+fname_ecc)
                ecc_map = ecc_img.get_fdata()
                sigma_img = nib.load(data_dir+fname_sigma)
                sigma_map = sigma_img.get_fdata()                
                aparc_img = nib.load(data_dir+fname_aparc)
                aparc_roi = aparc_img.get_fdata()
                label_img = nib.load(data_dir+fname_label)
                label_map = label_img.get_fdata()

                # compute valid voxels
                dot = (ecc_map * polar_map)
                good_coords = np.asarray(np.where(dot != 0.0))

                # filter gm per hemisphere
                cs_coords_rh = np.where(aparc_roi == 1021)
                cs_coords_lh = np.where(aparc_roi == 2021)
                gm_coords_rh = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
                gm_coords_lh = np.where(aparc_roi > 2000)
                xl,yl,zl = get_xyz(gm_coords_lh)
                xr,yr,zr = get_xyz(gm_coords_rh)
                GM_LH = np.array([xl,yl,zl]).T
                GM_RH = np.array([xr,yr,zr]).T

                # extract labels
                V1_coords_rh = np.asarray(np.where(label_map == 1))
                V1_coords_lh = np.asarray(np.where(label_map == 1))
                V2_coords_rh = np.asarray(np.where(label_map == 2))
                V2_coords_lh = np.asarray(np.where(label_map == 2))
                V3_coords_rh = np.asarray(np.where(label_map == 3))
                V3_coords_lh = np.asarray(np.where(label_map == 3))

                # divide V1 coords per hemisphere
                good_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                good_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V1_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V1_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V2_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V2_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                V3_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                V3_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T           

                # find center of left and right calcarine sulci
                median_lh = [np.median(cs_coords_lh[0][:]), np.median(cs_coords_lh[1][:]), np.median(cs_coords_lh[2][:])]
                median_rh = [np.median(cs_coords_rh[0][:]), np.median(cs_coords_rh[1][:]), np.median(cs_coords_rh[2][:])]

                # get GM mask and compute dorsal/posterior planes
                gm_mask = np.where(aparc_roi != 0)
                print('target: ', ftarget)
                print('loss: ', floss)
                print('a,b,c: ', a,b,c)

                # apply optimization to each hemisphere
                for gm_mask, hem, start_location, good_coords, good_coords_V1, good_coords_V2, good_coords_V3, dim2 in zip([GM_LH, GM_RH], ['LH', 'RH'], [median_lh, median_rh], [good_coords_lh, good_coords_rh], [V1_coords_lh, V1_coords_rh], [V2_coords_lh, V2_coords_rh], [V3_coords_lh, V3_coords_rh], [dim2_lh, dim2_rh]):        
                    
                    # check if already done
                    data_id = str(s) + '_' + str(hem) + '_V1_n1000_1x10_' + floss + '_' + str(thresh) + '_' + ftarget + '_spacing_' + str(spacing_x) + 'mm'
                    fname = output_folder + data_id + '.pkl'
                    if os.path.exists(fname):
                        print(str(s), ' ', str(hem), ' ',  str(ftarget), ' ', str(floss), ' already processed.')
                    else:
                        dimensions = [dim1, dim2, dim3, dim4]

                        # Define f function as closure to capture local variables
                        @use_named_args(dimensions=dimensions)
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
                            * Ultimately, the function returns the calculated cost used by the bayesopt algorithm.
                            """
                            
                            penalty = 0.25
                            new_angle = (float(alpha), float(beta), 0)    
                            
                            # create grid
                            orig_grid = create_grid_v2(
                                start_location,
                                shank_length=shank_length,
                                n_x=grid_nx, n_y=grid_ny, n_z=grid_nz,
                                spacing_x=spacing_x, spacing_y=spacing_y,
                                offset_from_origin=0
                            )
                            
                            # implanting grid
                            _, contacts_xyz_moved, _, _, _, _, _, _, grid_valid = implant_grid(gm_mask, orig_grid, start_location, new_angle, offset_from_base)

                            # get angle, ecc and rfsize for contactpoints (phosphenes[0-2][:] 0 angle x 1 ecc x 2 rfsize)    
                            phosphenes = get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
                            phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1, polar_map, ecc_map, sigma_map)
                            phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2, polar_map, ecc_map, sigma_map)   
                            phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3, polar_map, ecc_map, sigma_map)
                            
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
                            par1 = 1.0 - (a * dice)

                            # compute yield -> should be 1 -> invert cost
                            grid_yield = get_yield(contacts_xyz_moved, good_coords)
                            par2 = 1.0 - (b * grid_yield)

                            # compute hellinger distance -> should be small -> keep cost
                            hell_d = hellinger_distance(phospheneMap.flatten(), target_density.flatten())    
                            
                            ## validations steps
                            if np.isnan(phospheneMap).any() or np.sum(phospheneMap) == 0:
                                par1 = 1
                                print('map is nan or 0')
                            
                            if np.isnan(hell_d) or np.isinf(hell_d):
                                par3 = 1
                                print('Hellington is nan or inf')
                            else:
                                par3 = c * hell_d
                            
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

                        # create initial point generator
                        lhs2 = cook_initial_point_generator("lhs", criterion="maximin")

                        # optimize
                        res = gp_minimize(f, x0=x0, dimensions=dimensions, n_jobs=-1, n_calls=num_calls, n_initial_points=num_initial_points, initial_point_generator=lhs2, callback=[custom_stopper])

                        # print results
                        # print('subject ', s, ' ', hem)
                        # print('best alpha:', res.x[0])
                        # print('best beta:',res.x[1])
                        # print('best offset_from_base:', res.x[2])
                        # print('best shank_length:',res.x[3])
                        print('subject ', s, ' ', hem, ', best alpha: ', res.x[0], ', best beta: ', res.x[1], ', best offset_from_base: ', res.x[2], ', best shank_length: ', res.x[3])
                        grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap = f_manual(
                            res.x[0], res.x[1], res.x[2], res.x[3],
                            good_coords, good_coords_V1, good_coords_V2, good_coords_V3,
                            target_density,
                            start_location, gm_mask, polar_map, ecc_map, sigma_map,
                            a, b, c)
                        print('best dice, yield, KL: ', dice, grid_yield, hell_d)

                        # show resulting binary phosphene map (reflects dice coefficient)
                        bin_thresh=np.percentile(phospheneMap, dc_percentile) #np.min(target_density) # bin_thesh determines size target
                        phospheneMap_bin = (phospheneMap > bin_thresh).astype(bool)
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.suptitle('binarized vs. raw phospheneMap')
                        plt.subplot(1,2,1).imshow(phospheneMap_bin, cmap = 'seismic', vmin=0, vmax=np.max(phospheneMap)/100)
                        plt.subplot(1,2,2).imshow(phospheneMap, cmap = 'seismic', vmin=0, vmax=np.max(phospheneMap)/100)
                        
                        # Save the plot instead of showing
                        plot_filename = output_folder + data_id + '_phosphene_map.png'
                        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                        plt.close()  # Close the figure to free memory
                        print('    max phospheneMap: ', np.max(phospheneMap))
                        print('    Plot saved as: ', plot_filename)        

                        # Saving the objects
                        data_id = str(s) + '_' + str(hem) + '_V1_n1000_1x10_' + floss + '_' + str(thresh) + '_' + ftarget + '_spacing_' + str(spacing_x) + 'mm'                    
                        fname = output_folder + data_id + '.pkl'
                        
                        # Save as pickle file (store only picklable summary instead of full OptimizeResult)
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
                                         good_coords,
                                         good_coords_V1,
                                         good_coords_V2,
                                         good_coords_V3,
                                         phosphenes,
                                         phosphenes_V1,
                                         phosphenes_V2,
                                         phosphenes_V3], file, protocol=-1)
                        
                        # Save as txt file for human readability
                        txt_filename = output_folder + data_id + '.txt'
                        with open(txt_filename, 'w') as file:
                            file.write(f"Subject: {s}\n")
                            file.write(f"Hemisphere: {hem}\n")
                            file.write(f"Target: {ftarget}\n")
                            file.write(f"Loss: {floss}\n")
                            file.write(f"Threshold: {thresh}\n")
                            file.write(f"Best parameters:\n")
                            file.write(f"  Alpha: {res.x[0]}\n")
                            file.write(f"  Beta: {res.x[1]}\n")
                            file.write(f"  Offset from base: {res.x[2]}\n")
                            file.write(f"  Shank length: {res.x[3]}\n")
                            file.write(f"Results:\n")
                            file.write(f"  Grid valid: {grid_valid}\n")
                            file.write(f"  Dice coefficient: {dice:.6f}\n")
                            file.write(f"  Hellinger distance: {hell_d:.6f}\n")
                            file.write(f"  Grid yield: {grid_yield:.6f}\n")
                            file.write(f"  Max phospheneMap: {np.max(phospheneMap):.6f}\n")
                            file.write(f"  Number of contact points: {len(contacts_xyz_moved)}\n")
                            file.write(f"  Number of V1 coordinates: {len(good_coords_V1)}\n")
                            file.write(f"  Number of V2 coordinates: {len(good_coords_V2)}\n")
                            file.write(f"  Number of V3 coordinates: {len(good_coords_V3)}\n")
                        
                        print(f"    Results saved as: {fname} and {txt_filename}")
    
    # Calculate and print total execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Convert to hours, minutes, seconds for better readability
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*60)
    print(f"TOTAL EXECUTION TIME: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total seconds: {total_time:.2f}")
    print("="*60)

def compare_spacing():
    """
    스페이싱 1.0mm와 0.2mm를 비교하는 함수
    기존 코드를 최대한 그대로 두면서 두 가지 스페이싱으로 최적화를 실행하고 결과를 비교
    """
    # 비교할 스페이싱 값들
    spacing_values = [1.0, 0.2]
    spacing_names = ['1mm', '0.2mm']
    
    # 결과 저장을 위한 폴더 설정
    base_output_folder = "C:/Users/user/YongtaeC/vimplant0812/data/"
    
    # 각 스페이싱에 대해 최적화 실행
    results = {}
    
    for spacing_val, spacing_name in zip(spacing_values, spacing_names):
        print(f"\n{'='*60}")
        print(f"스페이싱 {spacing_name} ({spacing_val}mm) 최적화 시작")
        print(f"{'='*60}")
        
        # 스페이싱 값 설정
        global spacing_x, spacing_y, spacing_along_xy
        spacing_x = spacing_val
        spacing_y = spacing_val
        spacing_along_xy = spacing_val
        
        # 출력 폴더 설정
        subject_id = subj_list[0]  # 현재 설정된 서브젝트
        output_folder = f"{base_output_folder}0920_spacing_{subject_id}/spacing_{spacing_name}/"
        os.makedirs(output_folder, exist_ok=True)
        
        # 기존 main 함수의 로직을 여기서 실행
        start_time = time.time()
        
        # 파일 이름들
        fname_ang = 'inferred_angle.mgz'
        fname_ecc = 'inferred_eccen.mgz'
        fname_sigma = 'inferred_sigma.mgz'
        fname_anat = 'T1.mgz'
        fname_aparc = 'aparc+aseg.mgz'
        fname_label = 'inferred_varea.mgz'
        
        # 베타 각도 제약 설정
        dim2_lh = Integer(name='beta', low=-15, high=110)
        dim2_rh = Integer(name='beta', low=-110, high=15)
        
        spacing_results = {}
        
        # 타겟 맵과 손실 함수 조합에 대해 루프
        for target_density, ftarget in zip(targ_comb, targ_names):
            for (a, b, c), floss in zip(loss_comb, loss_names):
                # 타겟 정규화
                target_density_copy = target_density.copy()
                target_density_copy /= target_density_copy.max()
                target_density_copy /= target_density_copy.sum()
                
                bin_thresh = np.percentile(target_density_copy, dc_percentile)
                target_density_bin = (target_density_copy > bin_thresh).astype(bool)
                
                for s in subj_list:
                    data_dir = datafolder
                    
                    if s == 'fsaverage':
                        data_dir = datafolder + str(s) + '/mri/'
                    
                    # 맵 로드
                    ang_img = nib.load(data_dir+fname_ang)
                    polar_map = ang_img.get_fdata()
                    ecc_img = nib.load(data_dir+fname_ecc)
                    ecc_map = ecc_img.get_fdata()
                    sigma_img = nib.load(data_dir+fname_sigma)
                    sigma_map = sigma_img.get_fdata()
                    aparc_img = nib.load(data_dir+fname_aparc)
                    aparc_roi = aparc_img.get_fdata()
                    label_img = nib.load(data_dir+fname_label)
                    label_map = label_img.get_fdata()
                    
                    # 유효한 복셀 계산
                    dot = (ecc_map * polar_map)
                    good_coords = np.asarray(np.where(dot != 0.0))
                    
                    # 반구별 GM 필터링
                    cs_coords_rh = np.where(aparc_roi == 1021)
                    cs_coords_lh = np.where(aparc_roi == 2021)
                    gm_coords_rh = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
                    gm_coords_lh = np.where(aparc_roi > 2000)
                    xl,yl,zl = get_xyz(gm_coords_lh)
                    xr,yr,zr = get_xyz(gm_coords_rh)
                    GM_LH = np.array([xl,yl,zl]).T
                    GM_RH = np.array([xr,yr,zr]).T
                    
                    # 라벨 추출
                    V1_coords_rh = np.asarray(np.where(label_map == 1))
                    V1_coords_lh = np.asarray(np.where(label_map == 1))
                    V2_coords_rh = np.asarray(np.where(label_map == 2))
                    V2_coords_lh = np.asarray(np.where(label_map == 2))
                    V3_coords_rh = np.asarray(np.where(label_map == 3))
                    V3_coords_lh = np.asarray(np.where(label_map == 3))
                    
                    # 반구별 좌표 분할
                    good_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                    good_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(good_coords).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                    V1_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                    V1_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V1_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                    V2_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                    V2_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V2_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                    V3_coords_lh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_lh).T) & set(tuple(x) for x in np.round(gm_coords_lh).T)]).T
                    V3_coords_rh = np.array([x for x in set(tuple(x) for x in np.round(V3_coords_rh).T) & set(tuple(x) for x in np.round(gm_coords_rh).T)]).T
                    
                    # 좌우 칼카린 고랑 중심 찾기
                    median_lh = [np.median(cs_coords_lh[0][:]), np.median(cs_coords_lh[1][:]), np.median(cs_coords_lh[2][:])]
                    median_rh = [np.median(cs_coords_rh[0][:]), np.median(cs_coords_rh[1][:]), np.median(cs_coords_rh[2][:])]
                    
                    # GM 마스크 및 등쪽/후쪽 평면 계산
                    gm_mask = np.where(aparc_roi != 0)
                    
                    # 각 반구에 대해 최적화 적용
                    for gm_mask, hem, start_location, good_coords, good_coords_V1, good_coords_V2, good_coords_V3, dim2 in zip([GM_LH, GM_RH], ['LH', 'RH'], [median_lh, median_rh], [good_coords_lh, good_coords_rh], [V1_coords_lh, V1_coords_rh], [V2_coords_lh, V2_coords_rh], [V3_coords_lh, V3_coords_rh], [dim2_lh, dim2_rh]):
                        
                        dimensions = [dim1, dim2, dim3, dim4]
                        
                        # f 함수 정의 (클로저로 지역 변수 캡처)
                        @use_named_args(dimensions=dimensions)
                        def f(alpha, beta, offset_from_base, shank_length):
                            penalty = 0.25
                            new_angle = (float(alpha), float(beta), 0)
                            
                            # 그리드 생성
                            orig_grid = create_grid_v2(
                                start_location,
                                shank_length=shank_length,
                                n_x=grid_nx, n_y=grid_ny, n_z=grid_nz,
                                spacing_x=spacing_x, spacing_y=spacing_y,
                                offset_from_origin=0
                            )
                            
                            # 그리드 이식
                            _, contacts_xyz_moved, _, _, _, _, _, _, grid_valid = implant_grid(gm_mask, orig_grid, start_location, new_angle, offset_from_base)
                            
                            # 접촉점에 대한 각도, 편심, RF 크기 얻기
                            phosphenes = get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
                            phosphenes_V1 = get_phosphenes(contacts_xyz_moved, good_coords_V1, polar_map, ecc_map, sigma_map)
                            phosphenes_V2 = get_phosphenes(contacts_xyz_moved, good_coords_V2, polar_map, ecc_map, sigma_map)
                            phosphenes_V3 = get_phosphenes(contacts_xyz_moved, good_coords_V3, polar_map, ecc_map, sigma_map)
                            
                            # 역피질 확대 (도/mm 조직)
                            M = 1 / get_cortical_magnification(phosphenes_V1[:,1], cort_mag_model)
                            spread = cortical_spread(amp)
                            sizes = spread*M
                            sigmas = sizes / 2
                            
                            # CMF + 자극 진폭 기반 포스펀 크기
                            phosphenes_V1[:,2] = sigmas
                            
                            # 가우시안을 사용한 맵 생성
                            phospheneMap = np.zeros((WINDOWSIZE,WINDOWSIZE), 'float32')
                            phospheneMap = prf_to_phos(phospheneMap, phosphenes_V1, view_angle=view_angle, phSizeScale=1)
                            phospheneMap /= phospheneMap.max()
                            phospheneMap /= phospheneMap.sum()
                            
                            # 주사위 계수 계산
                            bin_thresh = np.percentile(target_density_copy, dc_percentile)
                            dice, im1, im2 = DC(target_density_copy, phospheneMap, bin_thresh)
                            par1 = 1.0 - (a * dice)
                            
                            # 수율 계산
                            grid_yield = get_yield(contacts_xyz_moved, good_coords)
                            par2 = 1.0 - (b * grid_yield)
                            
                            # 헬링거 거리 계산
                            hell_d = hellinger_distance(phospheneMap.flatten(), target_density_copy.flatten())
                            
                            # 검증 단계
                            if np.isnan(phospheneMap).any() or np.sum(phospheneMap) == 0:
                                par1 = 1
                                print('map is nan or 0')
                            
                            if np.isnan(hell_d) or np.isinf(hell_d):
                                par3 = 1
                                print('Hellington is nan or inf')
                            else:
                                par3 = c * hell_d
                            
                            # 비용 함수 결합
                            cost = par1 + par2 + par3
                            
                            # 일부 접촉점이 반구 밖에 있을 때 페널티 추가
                            if not grid_valid:
                                cost = par1 + penalty + par2 + penalty + par3 + penalty
                            
                            # 비용에 유효하지 않은 값이 있는지 확인
                            if np.isnan(cost) or np.isinf(cost):
                                cost = 3
                            
                            print('    ', "{:.2f}".format(cost), "{:.2f}".format(dice), "{:.2f}".format(grid_yield), "{:.2f}".format(par3), grid_valid)
                            return cost
                        
                        # 초기점 생성기 생성
                        lhs2 = cook_initial_point_generator("lhs", criterion="maximin")
                        
                        # 최적화
                        res = gp_minimize(f, x0=x0, dimensions=dimensions, n_jobs=-1, n_calls=num_calls, n_initial_points=num_initial_points, initial_point_generator=lhs2, callback=[custom_stopper])
                        
                        # 결과 출력
                        print('subject ', s, ' ', hem, ', best alpha: ', res.x[0], ', best beta: ', res.x[1], ', best offset_from_base: ', res.x[2], ', best shank_length: ', res.x[3])
                        
                        # 최적화된 파라미터로 수동 실행
                        grid_valid, dice, hell_d, grid_yield, phosphenes, phosphenes_V1, phosphenes_V2, phosphenes_V3, contacts_xyz_moved, phospheneMap = f_manual(
                            res.x[0], res.x[1], res.x[2], res.x[3],
                            good_coords, good_coords_V1, good_coords_V2, good_coords_V3,
                            target_density_copy,
                            start_location, gm_mask, polar_map, ecc_map, sigma_map,
                            a, b, c)
                        
                        print('best dice, yield, KL: ', dice, grid_yield, hell_d)
                        
                        # 결과 저장
                        data_id = f"{s}_{hem}_V1_n1000_1x10_{floss}_{thresh}_{ftarget}_spacing_{spacing_name}"
                        fname = output_folder + data_id + '.pkl'
                        
                        # 결과를 딕셔너리에 저장
                        spacing_results[f"{s}_{hem}_{ftarget}_{floss}"] = {
                            'spacing': spacing_val,
                            'spacing_name': spacing_name,
                            'subject': s,
                            'hemisphere': hem,
                            'target': ftarget,
                            'loss': floss,
                            'best_params': res.x,
                            'best_cost': res.fun,
                            'dice': dice,
                            'hellinger_distance': hell_d,
                            'grid_yield': grid_yield,
                            'grid_valid': grid_valid,
                            'phospheneMap': phospheneMap,
                            'contacts_xyz_moved': contacts_xyz_moved,
                            'phosphenes_V1': phosphenes_V1,
                            'optimization_result': res
                        }
                        
                        # 바이너리 포스펀 맵 표시
                        bin_thresh = np.percentile(phospheneMap, dc_percentile)
                        phospheneMap_bin = (phospheneMap > bin_thresh).astype(bool)
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.suptitle(f'Spacing {spacing_name} - Binarized vs Raw PhospheneMap')
                        plt.subplot(1,2,1).imshow(phospheneMap_bin, cmap='seismic', vmin=0, vmax=np.max(phospheneMap)/100)
                        plt.subplot(1,2,2).imshow(phospheneMap, cmap='seismic', vmin=0, vmax=np.max(phospheneMap)/100)
                        
                        # 플롯 저장
                        plot_filename = output_folder + data_id + '_phosphene_map.png'
                        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print('    max phospheneMap: ', np.max(phospheneMap))
                        print('    Plot saved as: ', plot_filename)
                        
                        # 피클 파일로 저장
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
                                         good_coords,
                                         good_coords_V1,
                                         good_coords_V2,
                                         good_coords_V3,
                                         phosphenes,
                                         phosphenes_V1,
                                         phosphenes_V2,
                                         phosphenes_V3], file, protocol=-1)
                        
                        # 텍스트 파일로 저장
                        txt_filename = output_folder + data_id + '.txt'
                        with open(txt_filename, 'w') as file:
                            file.write(f"Subject: {s}\n")
                            file.write(f"Hemisphere: {hem}\n")
                            file.write(f"Target: {ftarget}\n")
                            file.write(f"Loss: {floss}\n")
                            file.write(f"Spacing: {spacing_name} ({spacing_val}mm)\n")
                            file.write(f"Threshold: {thresh}\n")
                            file.write(f"Best parameters:\n")
                            file.write(f"  Alpha: {res.x[0]}\n")
                            file.write(f"  Beta: {res.x[1]}\n")
                            file.write(f"  Offset from base: {res.x[2]}\n")
                            file.write(f"  Shank length: {res.x[3]}\n")
                            file.write(f"Results:\n")
                            file.write(f"  Grid valid: {grid_valid}\n")
                            file.write(f"  Dice coefficient: {dice:.6f}\n")
                            file.write(f"  Hellinger distance: {hell_d:.6f}\n")
                            file.write(f"  Grid yield: {grid_yield:.6f}\n")
                            file.write(f"  Max phospheneMap: {np.max(phospheneMap):.6f}\n")
                            file.write(f"  Number of contact points: {len(contacts_xyz_moved)}\n")
                            file.write(f"  Number of V1 coordinates: {len(good_coords_V1)}\n")
                            file.write(f"  Number of V2 coordinates: {len(good_coords_V2)}\n")
                            file.write(f"  Number of V3 coordinates: {len(good_coords_V3)}\n")
                        
                        print(f"    Results saved as: {fname} and {txt_filename}")
        
        results[spacing_name] = spacing_results
        
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\n스페이싱 {spacing_name} 완료 시간: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # 결과 비교 및 시각화
    create_comparison_plots(results, base_output_folder, subject_id)
    
    return results

def create_comparison_plots(results, base_output_folder, subject_id):
    """
    스페이싱 비교 결과를 시각화하는 함수
    """
    comparison_folder = f"{base_output_folder}0920_spacing_{subject_id}/comparison/"
    os.makedirs(comparison_folder, exist_ok=True)
    
    # 각 타겟과 손실 함수 조합에 대해 비교 플롯 생성
    for spacing_name, spacing_results in results.items():
        for key, result in spacing_results.items():
            target_loss = key.split('_', 2)[2]  # target_loss 부분 추출
            
            # 같은 타겟과 손실 함수 조합의 결과들을 찾기
            matching_results = []
            for other_spacing_name, other_results in results.items():
                for other_key, other_result in other_results.items():
                    if other_key.split('_', 2)[2] == target_loss:
                        matching_results.append(other_result)
            
            if len(matching_results) == 2:  # 두 스페이싱 결과가 모두 있을 때
                create_single_comparison_plot(matching_results, comparison_folder, target_loss)

def create_single_comparison_plot(results, comparison_folder, target_loss):
    """
    단일 비교 플롯 생성
    """
    if len(results) != 2:
        return
    
    # 결과 정렬 (스페이싱 순서대로)
    results.sort(key=lambda x: x['spacing'])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Spacing Comparison: {target_loss}', fontsize=16)
    
    # 각 스페이싱에 대해 플롯
    for i, result in enumerate(results):
        spacing_name = result['spacing_name']
        phospheneMap = result['phospheneMap']
        
        # 원본 포스펀 맵
        axes[i, 0].imshow(phospheneMap, cmap='seismic', vmin=0, vmax=np.max(phospheneMap)/100)
        axes[i, 0].set_title(f'{spacing_name} - Raw PhospheneMap')
        axes[i, 0].axis('off')
        
        # 바이너리 포스펀 맵
        bin_thresh = np.percentile(phospheneMap, dc_percentile)
        phospheneMap_bin = (phospheneMap > bin_thresh).astype(bool)
        axes[i, 1].imshow(phospheneMap_bin, cmap='seismic', vmin=0, vmax=1)
        axes[i, 1].set_title(f'{spacing_name} - Binary PhospheneMap')
        axes[i, 1].axis('off')
        
        # 접촉점 분포 (3D 투영)
        contacts = result['contacts_xyz_moved']
        if len(contacts) > 0:
            axes[i, 2].scatter(contacts[:, 0], contacts[:, 1], c=contacts[:, 2], cmap='viridis', alpha=0.7)
            axes[i, 2].set_title(f'{spacing_name} - Contact Points Distribution')
            axes[i, 2].set_xlabel('X (mm)')
            axes[i, 2].set_ylabel('Y (mm)')
        else:
            axes[i, 2].text(0.5, 0.5, 'No valid contacts', ha='center', va='center', transform=axes[i, 2].transAxes)
            axes[i, 2].set_title(f'{spacing_name} - No Valid Contacts')
    
    plt.tight_layout()
    
    # 비교 플롯 저장
    comparison_filename = comparison_folder + f'spacing_comparison_{target_loss}.png'
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved as: {comparison_filename}")
    
    # 수치 비교 요약 생성
    create_comparison_summary(results, comparison_folder, target_loss)

def create_comparison_summary(results, comparison_folder, target_loss):
    """
    수치 비교 요약 생성
    """
    summary_filename = comparison_folder + f'spacing_comparison_summary_{target_loss}.txt'
    
    with open(summary_filename, 'w') as file:
        file.write(f"Spacing Comparison Summary: {target_loss}\n")
        file.write("="*50 + "\n\n")
        
        for result in results:
            file.write(f"Spacing: {result['spacing_name']} ({result['spacing']}mm)\n")
            file.write(f"Subject: {result['subject']}\n")
            file.write(f"Hemisphere: {result['hemisphere']}\n")
            file.write(f"Target: {result['target']}\n")
            file.write(f"Loss: {result['loss']}\n")
            file.write(f"Best Parameters:\n")
            file.write(f"  Alpha: {result['best_params'][0]}\n")
            file.write(f"  Beta: {result['best_params'][1]}\n")
            file.write(f"  Offset from base: {result['best_params'][2]}\n")
            file.write(f"  Shank length: {result['best_params'][3]}\n")
            file.write(f"Results:\n")
            file.write(f"  Best cost: {result['best_cost']:.6f}\n")
            file.write(f"  Dice coefficient: {result['dice']:.6f}\n")
            file.write(f"  Hellinger distance: {result['hellinger_distance']:.6f}\n")
            file.write(f"  Grid yield: {result['grid_yield']:.6f}\n")
            file.write(f"  Grid valid: {result['grid_valid']}\n")
            file.write(f"  Number of contacts: {len(result['contacts_xyz_moved'])}\n")
            file.write(f"  Max phospheneMap: {np.max(result['phospheneMap']):.6f}\n")
            file.write("\n" + "-"*30 + "\n\n")
        
        # 개선 정도 계산
        if len(results) == 2:
            result1, result2 = results
            file.write("Improvement Analysis:\n")
            file.write(f"Dice coefficient improvement: {result2['dice'] - result1['dice']:.6f}\n")
            file.write(f"Hellinger distance improvement: {result1['hellinger_distance'] - result2['hellinger_distance']:.6f}\n")
            file.write(f"Grid yield improvement: {result2['grid_yield'] - result1['grid_yield']:.6f}\n")
            file.write(f"Cost improvement: {result1['best_cost'] - result2['best_cost']:.6f}\n")
    
    print(f"Comparison summary saved as: {summary_filename}")

def visualize_spacing_results(spacing_values, subject_id, base_output_folder):
    """
    각 스페이싱에 대한 Loss, DC, Y, HD를 꺽은선그래프로 시각화
    - 우선 pkl 파일에서 수치( loss, dice, yield, hellinger_distance )를 읽음
    - 필요한 경우 txt 파일을 보조적으로 사용
    """
    import glob
    
    # 결과 저장 폴더 설정
    visualization_folder = f"{base_output_folder}0930_spacing_{subject_id}/visualization/"
    os.makedirs(visualization_folder, exist_ok=True)
    
    # 각 스페이싱별 결과 수집
    results_data = {
        'spacing': [],
        'loss': [],
        'dice': [],
        'yield': [],
        'hellinger_distance': []
    }
    
    # 각 스페이싱 폴더에서 결과 파일 찾기
    for spacing_val in spacing_values:
        spacing_folder = f"{base_output_folder}0930_spacing_{subject_id}/spacing_{spacing_val}mm/"
        
        # 1) pkl 우선 탐색
        pkl_files = glob.glob(f"{spacing_folder}*.pkl")
        picked = False
        for pkl_file in pkl_files:
            filename = os.path.basename(pkl_file)
            if 'LH' in filename and 'dice-yield-HD' in filename and 'targ-full' in filename:
                try:
                    with open(pkl_file, 'rb') as f:
                        obj = pickle.load(f)
                        # 저장 순서에 맞춰 언패킹
                        # [res_slim, grid_valid, dice, hell_d, grid_yield, contacts_xyz_moved, good_coords, ...]
                        res_slim = obj[0]
                        dice_val = obj[2]
                        hellinger_val = obj[3]
                        yield_val = obj[4]
                        loss_val = res_slim.get('fun', None) if isinstance(res_slim, dict) else None
                        if all(val is not None for val in [dice_val, hellinger_val, yield_val, loss_val]):
                            results_data['spacing'].append(spacing_val)
                            results_data['loss'].append(float(loss_val))
                            results_data['dice'].append(float(dice_val))
                            results_data['yield'].append(float(yield_val))
                            results_data['hellinger_distance'].append(float(hellinger_val))
                            picked = True
                            break
                except Exception as e:
                    print(f"Error reading {pkl_file}: {e}")
                    continue
        if picked:
            continue
        
        # 2) 보조적으로 txt 탐색 (loss가 없을 수 있음)
        txt_files = glob.glob(f"{spacing_folder}*.txt")
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    content = f.read()
                    filename = os.path.basename(txt_file)
                    if 'LH' in filename and 'dice-yield-HD' in filename and 'targ-full' in filename:
                        lines = content.split('\n')
                        dice_val = None
                        hellinger_val = None
                        yield_val = None
                        loss_val = None
                        for line in lines:
                            if 'Dice coefficient:' in line:
                                dice_val = float(line.split(':')[1].strip())
                            elif 'Hellinger distance:' in line:
                                hellinger_val = float(line.split(':')[1].strip())
                            elif 'Grid yield:' in line:
                                yield_val = float(line.split(':')[1].strip())
                            elif 'Best cost:' in line:
                                # 있을 수도, 없을 수도 있음
                                try:
                                    loss_val = float(line.split(':')[1].strip())
                                except:
                                    loss_val = None
                        if all(val is not None for val in [dice_val, hellinger_val, yield_val]):
                            results_data['spacing'].append(spacing_val)
                            # loss가 없으면 NaN으로 채움
                            results_data['loss'].append(float(loss_val) if loss_val is not None else np.nan)
                            results_data['dice'].append(float(dice_val))
                            results_data['yield'].append(float(yield_val))
                            results_data['hellinger_distance'].append(float(hellinger_val))
                            break
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
                continue
    
    # 데이터가 없으면 경고
    if not results_data['spacing']:
        print("시각화할 데이터를 찾을 수 없습니다.")
        return
    
    # 스페이싱 순서대로 정렬
    sorted_indices = sorted(range(len(results_data['spacing'])), key=lambda i: results_data['spacing'][i])
    for key in results_data:
        results_data[key] = [results_data[key][i] for i in sorted_indices]
    
    # 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Spacing Analysis Results - Subject {subject_id}', fontsize=16, fontweight='bold')
    
    # Loss 그래프
    axes[0, 0].plot(results_data['spacing'], results_data['loss'], 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 0].set_title('Loss (Cost Function)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Spacing (mm)', fontsize=12)
    axes[0, 0].set_ylabel('Loss Value', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(results_data['spacing'])
    
    # Dice Coefficient 그래프
    axes[0, 1].plot(results_data['spacing'], results_data['dice'], 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Spacing (mm)', fontsize=12)
    axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(results_data['spacing'])
    axes[0, 1].set_ylim(0, 1)
    
    # Yield 그래프
    axes[1, 0].plot(results_data['spacing'], results_data['yield'], 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_title('Grid Yield', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Spacing (mm)', fontsize=12)
    axes[1, 0].set_ylabel('Yield Value', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(results_data['spacing'])
    axes[1, 0].set_ylim(0, 1)
    
    # Hellinger Distance 그래프
    axes[1, 1].plot(results_data['spacing'], results_data['hellinger_distance'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 1].set_title('Hellinger Distance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Spacing (mm)', fontsize=12)
    axes[1, 1].set_ylabel('Hellinger Distance', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(results_data['spacing'])
    
    plt.tight_layout()
    
    # 그래프 저장
    plot_filename = visualization_folder + f'spacing_analysis_{subject_id}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"시각화 그래프가 저장되었습니다: {plot_filename}")
    
    # 데이터 요약 저장
    summary_filename = visualization_folder + f'spacing_summary_{subject_id}.txt'
    with open(summary_filename, 'w') as f:
        f.write(f"Spacing Analysis Summary - Subject {subject_id}\n")
        f.write("="*50 + "\n\n")
        f.write("Spacing (mm)\tLoss\t\tDice\t\tYield\t\tHellinger Distance\n")
        f.write("-"*70 + "\n")
        
        for i in range(len(results_data['spacing'])):
            f.write(f"{results_data['spacing'][i]:.1f}\t\t"
                   f"{results_data['loss'][i]:.6f}\t"
                   f"{results_data['dice'][i]:.6f}\t"
                   f"{results_data['yield'][i]:.6f}\t"
                   f"{results_data['hellinger_distance'][i]:.6f}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("Analysis Notes:\n")
        f.write("- Loss: Lower is better\n")
        f.write("- Dice Coefficient: Higher is better (0-1)\n")
        f.write("- Yield: Higher is better (0-1)\n")
        f.write("- Hellinger Distance: Lower is better\n")
    
    print(f"데이터 요약이 저장되었습니다: {summary_filename}")
    
    # 최적 스페이싱 찾기 (loss가 NaN인 경우 제외)
    loss_array = np.array(results_data['loss'], dtype=float)
    valid_idx = np.where(~np.isnan(loss_array))[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmin(loss_array[valid_idx])]
        best_spacing = results_data['spacing'][best_idx]
        print(f"\n최적 스페이싱: {best_spacing}mm")
        print(f"  - Loss: {results_data['loss'][best_idx]:.6f}")
        print(f"  - Dice: {results_data['dice'][best_idx]:.6f}")
        print(f"  - Yield: {results_data['yield'][best_idx]:.6f}")
        print(f"  - Hellinger Distance: {results_data['hellinger_distance'][best_idx]:.6f}")
    else:
        print("Loss 값을 찾지 못해 최적 스페이싱을 결정할 수 없습니다. (그래프는 저장됨)")

if __name__ == "__main__":
    # 스페이싱 0.2부터 2.0까지 0.2 간격으로 실행
    spacing_values = [round(x, 1) for x in np.arange(0.2, 2.0 + 1e-9, 0.2)]
    
    for spacing_val in spacing_values:
        print(f"\n{'='*60}")
        print(f"스페이싱 {spacing_val}mm 실행 시작")
        print(f"{'='*60}")
        
        # 스페이싱 값 설정
        spacing_x = spacing_val
        spacing_y = spacing_val
        spacing_along_xy = spacing_val
        
        # 출력 폴더 설정
        subject_id = subj_list[0]  # 현재 설정된 서브젝트 (100610)
        base_output_folder = "C:/Users/user/YongtaeC/vimplant0812/data/"
        output_folder = f"{base_output_folder}0930_spacing_{subject_id}/spacing_{spacing_val}mm/"
        os.makedirs(output_folder, exist_ok=True)
        
        # 기존 main 함수 실행 (출력 폴더 지정)
        main(output_folder)
    
    print("\n" + "="*60)
    print("모든 스페이싱 실행 완료!")
    print("="*60)
    
    # 결과 시각화
    print("\n결과 시각화 시작...")
    visualize_spacing_results(spacing_values, subject_id, base_output_folder)
