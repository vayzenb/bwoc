import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/bwoc'
sys.path.append(curr_dir)

import os

import numpy as np
import pandas as pd

study ='bwoc'
cope = 5
task = 'toolloc'

vols = 341
tr = 1
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')

loc_data = f'/lab_data/behrmannlab/vlad/hemispace'
erd_data = f'/lab_data/behrmannlab/vlad/docnet'

study_dir = f'/lab_data/behrmannlab/vlad/{study}'
scratch_dir =f'/lab_data/behrmannlab/scratch/vlad/{study}'