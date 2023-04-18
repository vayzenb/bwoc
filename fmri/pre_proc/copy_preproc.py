import os
import shutil
import bwoc_params as params

runs = [1,2]
for sub in params.sub_info['loc_sub']:
    og_sub_dir = f'{params.loc_data}/{sub}/ses-01/fsl/toolloc'
    #create subject directory in bwoc derivatives
    sub_dir = f'{params.study_dir}/{sub}/derivatives/fsl/toolloc'
    os.makedirs(sub_dir, exist_ok=True)

    #create run dirs in bwoc derivatives
    for run in runs:
        run_dir = os.path.join(sub_dir, f'run-0{run}')
        os.makedirs(run_dir, exist_ok=True)


    break

