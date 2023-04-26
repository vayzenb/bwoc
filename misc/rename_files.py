#rename all folders in scratch that end with _z.nii.gz
curr_dir = '/user_data/vayzenbe/GitHub_Repos/bwoc'
import sys
sys.path.insert(0, curr_dir)
import os
import shutil
import bwoc_params as params
from glob import glob as glob

file_list = glob(f'{params.scratch_dir}/derivatives/fc/*_z.nii.gz')

for file in file_list:
    new_file = file.replace('_z.nii.gz', '_dist.nii.gz')
    shutil.move(file, new_file)

    #delete old file
    #os.remove(file)

