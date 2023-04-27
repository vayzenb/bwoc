# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/bwoc'
sys.path.insert(0, curr_dir)
import warnings
warnings.filterwarnings("ignore")
import resource

import time
import os
import gc
import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression

from nilearn.maskers import NiftiSpheresMasker, NiftiMasker

from nilearn import image, datasets
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball
import bwoc_params as params

from statsmodels.tsa.stattools import grangercausalitytests

print('libraries loaded...')



#load subj number and seed
#subj
loc_sub = sys.argv[1]
erd_sub = sys.argv[2]
#seed region
roi = sys.argv[3]

print(loc_sub,erd_sub, roi)
# %%
#setup directories
study ='docnet'

out_dir = f'{params.scratch_dir}/derivatives/efc'
os.makedirs(out_dir, exist_ok=True)
results_dir = f'/{curr_dir}/results'
exp = 'catmvpa'

sub_dir = f'{params.study_dir}/sub-{erd_sub}/ses-02/'

roi_dir = f'{params.study_dir}/sub-{loc_sub}/ses-01/derivatives/rois'
exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'

runs = list(range(1,9))

print(erd_sub, roi, runs)


whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_strucseg_periph.nii.gz')
affine = whole_brain_mask.affine
dimsize = whole_brain_mask.header.get_zooms()  #get dimenisions

# scan parameters
vols = 331
first_fix = 8

# threshold for PCA
pc_thresh = .9

clf = LinearRegression()
#train/test split in 6 and 2 runs
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)

"""
Setup searchlight
"""
print('Searchlight setup ...')
#set search light params

mask = image.get_data(whole_brain_mask) #the mask to search within


sl_rad = 2 #radius of searchlight sphere (in voxels)
max_blk_edge = 5 #how many blocks to send on each parallelized search
pool_size = 1 #number of cores to work on each search

voxels_proportion=1
shape = Ball




# %%

def create_ts_mask(train, test):
    """
    Create timeseries mask (i.e., a list of value)  that correspond to training and test runs
    """

    train_index = []
    test_index = []

    for tr in train:
        train_index = train_index + list(range((tr-1) * (vols-first_fix),((tr-1) * (vols-first_fix)) + (vols-first_fix)))

    for te in test:
        test_index = test_index + list(range((te-1) * (vols-first_fix),((te-1) * (vols-first_fix)) + (vols-first_fix)))

    return train_index, test_index


# %%
def gca(data, sl_mask, myrad, seed_ts):
    """
    Seed to sphere gca
    """
    
    # Pull out the data
    data4D = data[0]
    data4D = np.transpose(data4D.reshape(-1, data[0].shape[3]))
    #print('mvpd', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)

    

    #calcualte mean TS for target region
    target_ts = np.mean(data4D, axis=1)

    neural_ts= pd.DataFrame(columns = ['seed', 'target'])
    neural_ts['seed'] = np.squeeze(seed_ts)
    neural_ts['target'] = np.squeeze(target_ts)

    #extract F stat from granger causality test seed-> target
    gc_res_seed = grangercausalitytests(neural_ts[['target','seed']], 1, verbose=False)

    #extract F stat from granger causality test target-> seed
    gc_res_target = grangercausalitytests(neural_ts[['seed','target']], 1, verbose=False)
    
    #calc difference
    f_diff = gc_res_seed[1][0]['ssr_ftest'][0]-gc_res_target[1][0]['ssr_ftest'][0]


    return f_diff    



# %%
def load_data():
    print('Loading data...')

    all_runs = []
    for run in runs:
        print(run)

        curr_run = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz') #load data
        curr_run = image.get_data(image.clean_img(curr_run,standardize=True,mask_img=whole_brain_mask)) #standardize within mask and convert to numpy
        curr_run = curr_run[:,:,:,first_fix:] #remove first few fixation volumes

        all_runs.append(curr_run) #append to list


        del curr_run
        print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)

    print('data loaded..')

    print('concatenating data..')
    bold_vol = np.concatenate(np.array(all_runs),axis = 3) #compile into 4D
    del all_runs
    print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
    print('data concatenated...')
    gc.collect()



    return bold_vol



    # %%
def extract_seed_ts(bold_vol, coords):
    """
    extract all data from seed region
    """

    roi_masker = NiftiSpheresMasker([tuple(coords)], radius = 6)

    bold_vol = nib.Nifti1Image(bold_vol, affine)
    seed_time_series = roi_masker.fit_transform(bold_vol)


    phys = np.mean(seed_time_series, axis= 1)
    #phys = (phys - np.mean(phys)) / np.std(phys) #zscore
    phys = phys.reshape((phys.shape[0],1))
    

    print('Seed data extracted...')

    return phys

# %%


#load coords for current roi
roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')
curr_coords = roi_coords[(roi_coords['task'] ==params.task) & (roi_coords['roi'] ==roi)]


#load TS
bold_vol = load_data()

seed_ts = extract_seed_ts(bold_vol,curr_coords[['x','y','z']].values.tolist()[0])

# %%
#run searchlight
t1 = time.time()
print("Begin Searchlight", print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024))
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge, shape = shape) #setup the searchlight
print('Distribute', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.distribute([bold_vol], mask) #send the 4dimg and mask

print('Broadcast', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.broadcast(seed_ts) #send the relevant analysis vars
print('Run', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024, flush= True)
sl_result = sl.run_searchlight(gca, pool_size=pool_size)
print("End Searchlight\n", (time.time()-t1)/60)


# %%
sl_result = sl_result.astype('double')  # Convert the output into a precision format that can be used by other applications
sl_result[np.isnan(sl_result)] = 0  # Exchange nans with zero to ensure compatibility with other applications
sl_nii = nib.Nifti1Image(sl_result, affine)  # create the volume image
nib.save(sl_nii, f'{out_dir}/sub-{erd_sub}_{roi}_efc.nii.gz')  # Save the volume

