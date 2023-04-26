import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/bwoc'
sys.path.insert(0, curr_dir)
import pandas as pd
from nilearn import image, plotting,  glm
#from nilearn.glm import threshold_stats_img
import numpy as np
import numpy.linalg as npl

from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
import nibabel as nib


import os
import statsmodels.api as sm
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import matplotlib.pyplot as plt
import pdb
from scipy.stats import gamma
import warnings

import bwoc_params as params

warnings.filterwarnings('ignore')

#load sub and roi info from command line

loc_sub = sys.argv[1]
erd_sub = sys.argv[2]
roi = sys.argv[3]


'''exp info'''

out_dir = f'{params.scratch_dir}/derivatives/fc'
os.makedirs(out_dir, exist_ok=True)

cov_dir = f'{params.loc_data}'
results_dir = f'/{curr_dir}/results'
roi_suf = 'toolloc'

loc_task = 'toolloc'
erd_task = 'catmvpa'

#create runs 1 to 8
runs = list(range(1,9))



file_suf = ''


whole_brain_mask = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
mni = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
brain_masker = NiftiMasker(whole_brain_mask,
    smoothing_fwhm=0, standardize=True)


def extract_roi_sphere(img, coords):
    roi_masker = NiftiSpheresMasker([tuple(coords)], radius = 6)
    seed_time_series = roi_masker.fit_transform(img)

    inv_affine = npl.inv(img.affine)
    
    
    phys = np.mean(seed_time_series, axis= 1)
    #phys = (phys - np.mean(phys)) / np.std(phys) #zscore
    phys = phys.reshape((phys.shape[0],1))
    
    return phys

def calc_distance(img, coords):
    mni_img = nib.load(mni)
    inv_affine = npl.inv(img.affine)
    np_coords = image.coord_transform(int(coords[0]), int(coords[1]), int(coords[2]), inv_affine)

    blank_img = image.get_data(image.new_img_like(mni_img, np.zeros(mni_img.shape)))
    mni_mask_np = image.get_data(whole_brain_mask)

    for i in range(blank_img.shape[0]):
        for j in range(blank_img.shape[1]):
            for k in range(blank_img.shape[2]):
                #calculate distance between np_coords and current voxel
                dist = np.sqrt((i-np_coords[0])**2 + (j-np_coords[1])**2 + (k-np_coords[2])**2)
                if mni_mask_np[i,j,k] == 1:
                    #assign value to blank_img
                    blank_img[i,j,k] = dist
                else:
                    continue
    
    dist_img = nib.Nifti1Image(blank_img, mni_img.affine, mni_img.header)

    return dist_img

def conduct_fc(loc_sub, erd_sub, roi):

    print('Calculating fc for...', loc_sub,erd_sub, roi)
    erd_dir = f'{params.study_dir}/sub-{erd_sub}/ses-02/'
    loc_dir = f'{params.study_dir}/sub-{loc_sub}/ses-01/'
    roi_dir = f'{loc_dir}/derivatives/rois'
    exp_dir = f'{erd_dir}/derivatives/fsl/{erd_task}'

    roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')

    for tsk in ['toolloc']:

    
        curr_coords = roi_coords[(roi_coords['task'] ==tsk) & (roi_coords['roi'] ==roi)]

        filtered_list = []
        for run in runs:
            
            curr_run = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
            curr_run = image.clean_img(curr_run,standardize=True)
            filtered_list.append(curr_run)
            
        img4d = image.concat_imgs(filtered_list)

        #Extract mean whole brain TS
        whole_brain_ts = image.get_data(img4d)
        
        whole_brain_ts = np.mean(whole_brain_ts, axis = (0,1,2))

        confounds = pd.DataFrame(columns =['mean_signal'])
        confounds['mean_signal'] = whole_brain_ts
        
        phys = extract_roi_sphere(img4d,curr_coords[['x','y','z']].values.tolist()[0])
        dist_img = calc_distance(img4d,curr_coords[['x','y','z']].values.tolist()[0])
        
        #Extract TS values for whole brain
        brain_time_series = brain_masker.fit_transform(img4d,confounds=[confounds])
        
        #Extract distance values for whole brain
        dist_vals = brain_masker.fit_transform(dist_img)
        


        #Correlate seed to whole brain
        seed_to_voxel_correlations = (np.dot(brain_time_series.T, phys) /
                        phys.shape[0])
        print(erd_sub, roi, tsk, seed_to_voxel_correlations.max())
        
        #convert to fisher-z
        seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)

        #z-score correlations
        z_correlations = (seed_to_voxel_correlations - np.mean(seed_to_voxel_correlations)) / np.std(seed_to_voxel_correlations)

        #z-score distance values
        z_dist = (dist_vals - np.mean(dist_vals)) / np.std(dist_vals)


        #regress out distance values
        ols_model = sm.OLS(z_correlations, z_dist.T)
        #save residuals from regression
        ols_results = ols_model.fit()
        correlation_residuals = ols_results.resid

        #standardize residuals
        correlation_residuals = (correlation_residuals - np.mean(correlation_residuals)) / np.std(correlation_residuals)


        

        #transform correlation map back to brain
        seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)   

        #transform dist normalize correlations    
        dist_correlations_img = brain_masker.inverse_transform(correlation_residuals.T)

        mean_fc = seed_to_voxel_correlations_img
            
        nib.save(mean_fc, f'{out_dir}/sub-{erd_sub}_{roi}_{tsk}_fc.nii.gz')
        nib.save(dist_correlations_img, f'{out_dir}/sub-{erd_sub}_{roi}_{tsk}_fc_dist.nii.gz')





conduct_fc(loc_sub, erd_sub, roi)







        







