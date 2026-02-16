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
subj_list = [102311]

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
    orig_grid = create_grid(start_location, shank_length, n_contactpoints_shank, spacing_along_xy, offset_from_origin=0)

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
                            orig_grid = create_grid(start_location, shank_length, n_contactpoints_shank, spacing_along_xy, offset_from_origin=0)
                            
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

if __name__ == "__main__":
    main()
