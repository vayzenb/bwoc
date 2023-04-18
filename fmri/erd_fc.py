import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/bwoc'
sys.path.insert(0, curr_dir)
import pandas as pd
from nilearn import image, plotting, input_data, glm, maskers
#from nilearn.glm import threshold_stats_img
import numpy as np

from nilearn.input_data import NiftiMasker
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

out_dir = f'{params.study_dir}/derivatives/fc'
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
brain_masker = input_data.NiftiMasker(whole_brain_mask,
    smoothing_fwhm=0, standardize=True)


def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius = 6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis= 1)
    #phys = (phys - np.mean(phys)) / np.std(phys) #zscore
    phys = phys.reshape((phys.shape[0],1))
    
    return phys

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
        
        phys = extract_roi_sphere(img4d,curr_coords[['x','y','z']].values.tolist()[0])
        #load behavioral data
        #CONVOLE TO HRF
        

        brain_time_series = brain_masker.fit_transform(img4d)

        #Correlate interaction term to TS for vox in the brain
        seed_to_voxel_correlations = (np.dot(brain_time_series.T, phys) /
                        phys.shape[0])
        print(erd_sub, roi, tsk, seed_to_voxel_correlations.max())
        
        seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
        #transform correlation map back to brain
        seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
        
        

        mean_fc = seed_to_voxel_correlations_img
            
        nib.save(mean_fc, f'{out_dir}/sub-{erd_sub}_{roi}_{tsk}_fc.nii.gz')





conduct_fc(loc_sub, erd_sub, roi)







        







