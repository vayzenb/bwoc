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
exp = 'fc'
sub_list = params.sub_info[sub_cond+'_sub']

results_dir = f'/{curr_dir}/results'

rois = ['PPC', 'APC', 'LO', 'PFS']
roi_suf = '_toolloc'
# create left and right versinos of each roi
rois = ['l' + roi + roi_suf for roi in rois] + ['r' + roi + roi_suf for roi in rois]

file_suf = '_dist'

out_dir = f'{params.scratch_dir}/derivatives/fc'

whole_brain_mask = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_strucseg_periph.nii.gz'
#whole_brain_mask = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'

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
        if os.path.exists(f'{out_dir}/sub-{sub}_{roi}_fc.nii.gz'):
            
            curr_img = image.load_img(f'{out_dir}/sub-{sub}_{roi}_fc{file_suf}.nii.gz')


            #z_img = image.clean_img(curr_img, standardize=True)

            fc_img_z.append(curr_img)
            fc_img.append(curr_img)

    design_matrix = pd.DataFrame([1] * len(fc_img_z),
                            columns=['intercept'])
    final_img= second_level_model.fit(fc_img_z, design_matrix= design_matrix)
    z_map = final_img.compute_contrast(output_type='z_score')

    #mean images
    mean_img = image.mean_img(fc_img)

    thresh_val = glm.threshold_stats_img(z_map,alpha=alpha,height_control='fdr', cluster_threshold = 4,mask_img= whole_brain_mask)
    
    os.makedirs(f'{results_dir}/{exp}', exist_ok=True)
    nib.save(z_map, f'{results_dir}/{exp}/{roi}_{exp}_z{file_suf}.nii.gz')

    nib.save(mean_img, f'{results_dir}/{exp}/{roi}_{exp}_corr{file_suf}.nii.gz')
    print(f'{roi}', thresh_val[1])
