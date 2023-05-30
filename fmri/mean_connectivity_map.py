"""Create avg connectivity map"""

import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/bwoc'
sys.path.insert(0, curr_dir)
import pandas as pd
from nilearn import image, plotting, input_data, glm
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

sub_cond = 'erd'
exp = 'efc'
sub_list = params.sub_info[sub_cond+'_sub']

results_dir = f'/{curr_dir}/results/{exp}'
os.makedirs(results_dir, exist_ok=True)

rois = ['PPC', 'APC', 'LO', 'PFS']
roi_suf = ''
# create left and right versinos of each roi
rois = ['l' + roi + roi_suf for roi in rois] + ['r' + roi + roi_suf for roi in rois]

analysis_type = 'efc'
file_suf = ''


out_dir = f'{params.scratch_dir}/derivatives/{analysis_type}'

#whole_brain_mask = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_strucseg_periph.nii.gz'
whole_brain_mask = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
whole_brain_mask = image.load_img(f'{params.study_dir}/derivatives/rois/gm_mask.nii.gz')
"""
run 2nd level model on each first level
"""
second_level_model = glm.second_level.SecondLevelModel(smoothing_fwhm=4)
alpha = .05


for roi in rois:
    print(roi)
    fc_img = []
    fc_img_z = []
    for sub in sub_list:
        #check if file exists
        np_mask = image.get_data(whole_brain_mask)
        if os.path.exists(f'{out_dir}/sub-{sub}_{roi}_{analysis_type}{file_suf}.nii.gz'):
            
            curr_img = image.load_img(f'{out_dir}/sub-{sub}_{roi}_{analysis_type}{file_suf}.nii.gz')
            affine = curr_img.affine
            

            np_img = image.get_data(curr_img)

            #mask out non-gm voxels
            np_img[np_mask == 0] = 0

            #convert back to nifti
            z_img = nib.Nifti1Image(np_img, affine)

            #z_img = image.clean_img(curr_img, mask_img =whole_brain_mask)

            fc_img_z.append(z_img)
            fc_img.append(curr_img)
            

    design_matrix = pd.DataFrame([1] * len(fc_img_z),
                            columns=['intercept'])
    final_img= second_level_model.fit(fc_img_z, design_matrix= design_matrix)
    z_map = final_img.compute_contrast(output_type='z_score')

    #mean images
    mean_img = image.mean_img(fc_img)

    thresh_val = glm.threshold_stats_img(z_map,alpha=alpha,height_control='fdr', cluster_threshold = 4,mask_img= whole_brain_mask)
    
    
    nib.save(z_map, f'{results_dir}/{roi}_{exp}_z{file_suf}.nii.gz')

    nib.save(mean_img, f'{results_dir}/{roi}_{exp}_mean{file_suf}.nii.gz')
    print(f'{roi}', thresh_val[1])
