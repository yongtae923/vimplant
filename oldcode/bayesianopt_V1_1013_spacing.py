# -*- coding: utf-8 -*-
"""
Created on Wed May 6 2021

@authors: R. van Hoof & A. Lozano

Jupyter 노트북을 Python 스크립트로 변환
"""

import time
import itertools
from matplotlib.colors import Normalize
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
datafolder = "C:/Users/user/YongtaeC/vimplant0812/data/input/100610/"
outputfolder = "C:/Users/user/YongtaeC/vimplant0812/data/output/100610_py/"
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
              gvs.complete_gauss(windowsize=1000, fwhm=1200, radiusLow=0, radiusHigh=500, center=None, plotting=False)])
targ_names = (['targ-upper', 'targ-lower', 'targ-inner', 'targ-full'])

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

def main():
    # Start timing
    start_time = time.time()
    
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
                    data_id = str(s) + '_' + str(hem) + '_V1_n1000_1x10_' + floss + '_' + str(thresh) + '_' + ftarget
                    fname = outputfolder + data_id + '.pkl'
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
                        plot_filename = outputfolder + data_id + '_phosphene_map.png'
                        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                        plt.close()  # Close the figure to free memory
                        print('    max phospheneMap: ', np.max(phospheneMap))
                        print('    Plot saved as: ', plot_filename)        

                        # Saving the objects
                        data_id = str(s) + '_' + str(hem) + '_V1_n1000_1x10_' + floss + '_' + str(thresh) + '_' + ftarget                    
                        fname = outputfolder + data_id + '.pkl'
                        
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
                        txt_filename = outputfolder + data_id + '.txt'
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
    스페이싱을 0.2, 0.4, ..., 1.8mm로 변경하면서
    물리적 길이(10mm)는 동일하게 유지하고 채널 개수(n_x, n_y, n_z)를 spacing에 맞게 조정해 비교 실행.
    결과는 data/output/100610_spacing_1013/ 아래 spacing별로 저장.
    """
    # 비교할 스페이싱 값들 (요청: 0.5, 1.0, 1.5)
    spacing_values = [0.5, 1.0, 1.5]
    spacing_names = [f"{v:.1f}mm" for v in spacing_values]
    
    # 결과 저장을 위한 폴더 설정
    base_output_folder = "C:/Users/user/YongtaeC/vimplant0812/data/output/100610_spacing/"
    
    # 각 스페이싱에 대해 최적화 실행
    results = {}
    
    for spacing_val, spacing_name in zip(spacing_values, spacing_names):
        print(f"\n{'='*60}")
        print(f"스페이싱 {spacing_name} ({spacing_val}mm) 최적화 시작")
        print(f"{'='*60}")
        
        # 스페이싱/그리드/샹크 설정: 물리 길이 10mm 고정, 채널 수는 길이/spacing
        global spacing_x, spacing_y, spacing_along_xy, grid_nx, grid_ny, grid_nz, shank_length, n_contactpoints_shank
        spacing_x = spacing_val
        spacing_y = spacing_val
        spacing_along_xy = spacing_val
        shank_length = 10  # mm, 고정 길이
        n_per_axis = max(1, int(round(10.0 / spacing_val)))
        grid_nx = n_per_axis
        grid_ny = n_per_axis
        grid_nz = n_per_axis
        n_contactpoints_shank = grid_nz
        print(f"[CONFIG] spacing={spacing_val}mm -> grid={grid_nx}x{grid_ny}x{grid_nz}, shank_length={shank_length}mm", flush=True)
        
        # 출력 폴더 설정
        subject_id = subj_list[0]  # 현재 설정된 서브젝트
        output_folder = f"{base_output_folder}spacing_{spacing_name}_{grid_nx}x{grid_ny}x{grid_nz}/"
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
        print(f"[INFO] Using maps: {fname_ang}, {fname_ecc}, {fname_sigma}, {fname_aparc}, {fname_label}", flush=True)
        
        # 베타 각도 제약 설정
        dim2_lh = Integer(name='beta', low=-15, high=110)
        dim2_rh = Integer(name='beta', low=-110, high=15)
        
        spacing_results = {}
        
        # 타겟 맵과 손실 함수 조합에 대해 루프
        for target_density, ftarget in zip(targ_comb, targ_names):
            print(f"[TARGET] {ftarget} starting", flush=True)
            for (a, b, c), floss in zip(loss_comb, loss_names):
                print(f"  [LOSS] combo={floss} (a,b,c)=({a},{b},{c})", flush=True)
                # 타겟 정규화
                target_density_copy = target_density.copy()
                target_density_copy /= target_density_copy.max()
                target_density_copy /= target_density_copy.sum()
                
                bin_thresh = np.percentile(target_density_copy, dc_percentile)
                target_density_bin = (target_density_copy > bin_thresh).astype(bool)
                
                for s in subj_list:
                    print(f"    [SUBJECT] {s} loading", flush=True)
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
                    print(f"    [SUBJECT] {s} maps loaded", flush=True)
                    
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
                        print(f"      [HEM] {hem} optimization start", flush=True)
                        # 결과 존재 여부 선확인: 이미 결과가 있으면 스킵
                        data_id = f"{s}_{hem}_V1_n1000_{grid_nx}x{grid_ny}x{grid_nz}_{floss}_{thresh}_{ftarget}_spacing_{spacing_name}"
                        fname = output_folder + data_id + '.pkl'
                        txt_filename = output_folder + data_id + '.txt'
                        plot_filename = output_folder + data_id + '_phosphene_map.png'
                        if os.path.exists(fname) and os.path.exists(txt_filename) and os.path.exists(plot_filename):
                            print(f"      [SKIP] Exists: {data_id}", flush=True)
                            # 기존 결과를 로드하여 비교용 results에 반영
                            try:
                                with open(fname, 'rb') as file:
                                    loaded = pickle.load(file)
                                res_slim_loaded = loaded[0] if isinstance(loaded, (list, tuple)) and len(loaded) > 0 else {}
                                grid_valid_loaded = loaded[1] if len(loaded) > 1 else None
                                dice_loaded = loaded[2] if len(loaded) > 2 else None
                                hell_d_loaded = loaded[3] if len(loaded) > 3 else None
                                grid_yield_loaded = loaded[4] if len(loaded) > 4 else None
                                contacts_xyz_moved_loaded = loaded[5] if len(loaded) > 5 else np.array([])

                                # 최적 파라미터로 phospheneMap 재생성 (비교 플롯 용)
                                best_x = res_slim_loaded.get('x', None) if isinstance(res_slim_loaded, dict) else None
                                if best_x is not None:
                                    _grid_valid, _dice, _hell_d, _grid_yield, _phosphenes, _phos_V1, _phos_V2, _phos_V3, _contacts_xyz_moved, phospheneMap_loaded = f_manual(
                                        best_x[0], best_x[1], best_x[2], best_x[3],
                                        good_coords, good_coords_V1, good_coords_V2, good_coords_V3,
                                        target_density_copy,
                                        start_location, gm_mask, polar_map, ecc_map, sigma_map,
                                        a, b, c)
                                else:
                                    phospheneMap_loaded = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype='float32')

                                spacing_results[f"{s}_{hem}_{ftarget}_{floss}"] = {
                                    'spacing': spacing_val,
                                    'spacing_name': spacing_name,
                                    'subject': s,
                                    'hemisphere': hem,
                                    'target': ftarget,
                                    'loss': floss,
                                    'best_params': best_x,
                                    'best_cost': res_slim_loaded.get('fun', None) if isinstance(res_slim_loaded, dict) else None,
                                    'dice': dice_loaded,
                                    'hellinger_distance': hell_d_loaded,
                                    'grid_yield': grid_yield_loaded,
                                    'grid_valid': grid_valid_loaded,
                                    'phospheneMap': phospheneMap_loaded,
                                    'contacts_xyz_moved': contacts_xyz_moved_loaded,
                                    'phosphenes_V1': None,
                                    'optimization_result': None
                                }
                            except Exception as e:
                                print(f"      [SKIP-LOAD-ERROR] {data_id}: {e}", flush=True)
                            continue
                        
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
                        print(f"      [HEM] {hem} gp_minimize running...", flush=True)
                        res = gp_minimize(f, x0=x0, dimensions=dimensions, n_jobs=-1, n_calls=num_calls, n_initial_points=num_initial_points, initial_point_generator=lhs2, callback=[custom_stopper])
                        print(f"      [HEM] {hem} gp_minimize done", flush=True)
                        
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
                        data_id = f"{s}_{hem}_V1_n1000_{grid_nx}x{grid_ny}x{grid_nz}_{floss}_{thresh}_{ftarget}_spacing_{spacing_name}"
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
                        print(f"      [SAVE] Plot saved: {plot_filename}", flush=True)
                        
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
                        print(f"      [SAVE] Pickle saved: {fname}", flush=True)
                        
                        # 텍스트 파일로 저장
                        txt_filename = output_folder + data_id + '.txt'
                        with open(txt_filename, 'w') as file:
                            file.write(f"Subject: {s}\n")
                            file.write(f"Hemisphere: {hem}\n")
                            file.write(f"Target: {ftarget}\n")
                            file.write(f"Loss: {floss}\n")
                            file.write(f"Spacing: {spacing_name} ({spacing_val}mm)\n")
                            file.write(f"Grid: {grid_nx} x {grid_ny} x {grid_nz}\n")
                            file.write(f"Shank length: {shank_length} mm\n")
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
                        
                        print(f"      [SAVE] Summary saved: {txt_filename}", flush=True)
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
    print("[COMPARE] create_comparison_plots start", flush=True)
    print(f"[COMPARE] subject={subject_id}", flush=True)
    print(f"[COMPARE] spacings={list(results.keys())}", flush=True)
    comparison_folder = f"{base_output_folder}0920_spacing_{subject_id}/comparison/"
    os.makedirs(comparison_folder, exist_ok=True)
    print(f"[COMPARE] folder={comparison_folder}", flush=True)
    
    # 각 타겟과 손실 함수 조합에 대해 비교 플롯 생성
    # 타겟/손실 조합별로 스페이싱 페어 비교 생성
    # key 형식: f"{s}_{hem}_{ftarget}_{floss}"
    # target_loss 식별자는 key.split('_', 2)[2]
    # 1) 우선 모든 결과를 target_loss별로 모은다
    grouped = {}
    for spacing_name, spacing_results in results.items():
        print(f"[COMPARE] scanning spacing={spacing_name} cases={len(spacing_results)}", flush=True)
        for key, result in spacing_results.items():
            target_loss = key.split('_', 2)[2]
            grouped.setdefault(target_loss, []).append(result)

    # 2) 각 target_loss에 대해 스페이싱 쌍 조합을 생성하여 비교 플롯을 만든다
    for target_loss, result_list in grouped.items():
        # 스페이싱별로 정렬
        result_list_sorted = sorted(result_list, key=lambda x: x['spacing'])
        print(f"[COMPARE] target_loss={target_loss} total_results={len(result_list_sorted)}", flush=True)
        # 두 개씩 모든 조합
        for r1, r2 in itertools.combinations(result_list_sorted, 2):
            pair = [r1.copy(), r2.copy()]
            print(f"[COMPARE] plotting pair: {r1['spacing_name']} vs {r2['spacing_name']} for {target_loss}", flush=True)
            create_single_comparison_plot(pair, comparison_folder, target_loss + f"_{r1['spacing_name']}_vs_{r2['spacing_name']}")

    # 3) 요청: LH/RH와 타깃별로 0.5/1.0/1.5mm를 한 장(2x3)으로 비교
    #    loss는 현재 하나이므로 무시, 동일 대상 기준으로 3개 스페이싱 수집
    grouped_hemi_target = {}
    for spacing_name, spacing_results in results.items():
        for key, result in spacing_results.items():
            hemi_target = f"{result['hemisphere']}_{result['target']}"
            grouped_hemi_target.setdefault(hemi_target, []).append(result)

    for hemi_target, result_list in grouped_hemi_target.items():
        # 0.5/1.0/1.5 정렬 및 3개 이상이면 상위 3개 선택
        result_list_sorted = sorted(result_list, key=lambda x: x['spacing'])
        # 동일 spacing이 중복될 수 있으므로 spacing_name으로 유니크 필터
        unique = {}
        for r in result_list_sorted:
            unique[r['spacing_name']] = r
        ordered = [unique.get(s) for s in ["0.5mm", "1.0mm", "1.5mm"] if unique.get(s) is not None]
        if len(ordered) == 3:
            print(f"[COMPARE-3] plotting triple for {hemi_target} with 0.5/1.0/1.5mm", flush=True)
            create_triple_comparison_plot(ordered, comparison_folder, hemi_target)
            # 메트릭 라인 플롯 생성
            create_metric_line_plot(ordered, comparison_folder, hemi_target)
        else:
            print(f"[COMPARE-3] skip {hemi_target} (need 3 spacings, have {len(ordered)})", flush=True)

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

def create_triple_comparison_plot(results, comparison_folder, hemi_target):
    """
    요청 포맷: 가로 3칸(0.5/1.0/1.5mm)의 raw phosphene map,
    세로 아래 3칸은 contact points distribution, 축/컬러스케일 동일.
    results: [r_0.5mm, r_1.0mm, r_1.5mm] (spacing_name 순서 보장)
    """
    if len(results) != 3:
        return

    # 공통 vmin/vmax 계산 (phosphene map)
    vmax = max(np.max(r['phospheneMap']) if r['phospheneMap'] is not None else 0 for r in results)
    if vmax == 0:
        vmax = 1.0
    vmin = 0

    # contact z 색상용 범위 (형태 표준화: Nx3)
    def to_n_by_3(arr):
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.ndim == 1 and a.size % 3 == 0:
            a = a.reshape(-1, 3)
        elif a.ndim == 2:
            if a.shape[1] == 3:
                pass
            elif a.shape[0] == 3:
                a = a.T
            else:
                return None
        else:
            return None
        return a

    contacts_std = []
    for r in results:
        a = to_n_by_3(r.get('contacts_xyz_moved'))
        if a is not None and a.shape[1] == 3 and a.size > 0:
            contacts_std.append(a)

    if len(contacts_std) > 0:
        concat = np.vstack(contacts_std)
        zmin, zmax = float(np.min(concat[:, 2])), float(np.max(concat[:, 2]))
        xmin, xmax = float(np.min(concat[:, 0])), float(np.max(concat[:, 0]))
        ymin, ymax = float(np.min(concat[:, 1])), float(np.max(concat[:, 1]))
    else:
        zmin, zmax = 0.0, 1.0
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{hemi_target} | Triple Spacing Comparison (Raw/Contacts)")

    # 위 행: raw phosphene
    for i, r in enumerate(results):
        pm = r['phospheneMap']
        axes[0, i].imshow(pm, cmap='seismic', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"{r['spacing_name']} - Raw")
        axes[0, i].axis('equal')
        axes[0, i].axis('off')

    # 아래 행: contact distribution (XY 산점도, 색은 Z)
    mappable_for_colorbar = None
    for i, r in enumerate(results):
        contacts = to_n_by_3(r.get('contacts_xyz_moved'))
        ax = axes[1, i]
        if contacts is not None and contacts.size > 0:
            sc = ax.scatter(contacts[:, 0], contacts[:, 1], c=contacts[:, 2], cmap='viridis', vmin=zmin, vmax=zmax, s=10, alpha=0.8)
            if mappable_for_colorbar is None:
                mappable_for_colorbar = sc
            ax.set_title(f"{r['spacing_name']} - Contacts")
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.text(0.5, 0.5, 'No valid contacts', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{r['spacing_name']} - No Contacts")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal', adjustable='box')

    # 공통 Z 색상바 추가 (오른쪽에 전용 축 생성)
    if mappable_for_colorbar is not None:
        # 본문 영역이 색상바 공간을 남기도록 rect 지정
        plt.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(mappable_for_colorbar, cax=cax)
        cbar.set_label('Depth (Z, mm)')
    else:
        plt.tight_layout()
    out_path = comparison_folder + f"triple_spacing_{hemi_target}_0.5_1.0_1.5.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[COMPARE-3] saved {out_path}", flush=True)

def create_metric_line_plot(results, comparison_folder, hemi_target):
    """
    LH_full 등 반구+타깃 단위로 x축 [0.5, 1.0, 1.5]에 대해
    4개 메트릭(loss, dice, yield, hd)을 한 그래프에 꺾은선으로 표시.
    results: [0.5mm, 1.0mm, 1.5mm] 순서의 dict 리스트
    """
    # x축과 정렬 보장
    order = {"0.5mm": 0, "1.0mm": 1, "1.5mm": 2}
    results_sorted = sorted(results, key=lambda r: order.get(r['spacing_name'], 99))
    x_vals = [0.5, 1.0, 1.5]

    loss_vals = [float(r.get('best_cost')) if r.get('best_cost') is not None else np.nan for r in results_sorted]
    dice_vals = [float(r.get('dice')) if r.get('dice') is not None else np.nan for r in results_sorted]
    yield_vals = [float(r.get('grid_yield')) if r.get('grid_yield') is not None else np.nan for r in results_sorted]
    hd_vals = [float(r.get('hellinger_distance')) if r.get('hellinger_distance') is not None else np.nan for r in results_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, loss_vals, marker='o', label='loss')
    plt.plot(x_vals, dice_vals, marker='o', label='dice')
    plt.plot(x_vals, yield_vals, marker='o', label='yield')
    plt.plot(x_vals, hd_vals, marker='o', label='hd')
    plt.xticks(x_vals, ["0.5", "1.0", "1.5"])
    plt.xlabel('Spacing (mm)')
    plt.ylabel('Metric value')
    plt.title(f'{hemi_target} | Metrics vs Spacing')
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = comparison_folder + f"metrics_lines_{hemi_target}_0.5_1.0_1.5.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[METRICS] saved {out_path}", flush=True)

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

if __name__ == "__main__":
    # 기존 main 함수 실행
    # main()
    
    # 스페이싱 비교 실행
    compare_spacing()
