"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/bwoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)

import subprocess
from glob import glob
import os
import time
import pdb
import bwoc_params as params

mem = 48
run_time = "3-00:00:00"

pause_time = 30 #how much time (minutes) to wait between jobs
pause_crit = 10 #how many jobs to do before pausing


sub_list = params.sub_info
rois = ['PPC', 'APC', 'LO', 'PFS']
#create left and right versinos of each roi
rois = ['l'+ roi for roi in rois] + ['r' + roi for roi in rois]


suf = ''

def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
# Submit job to cpu queue                
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
# Job memory request
#SBATCH --mem={mem}gb
# Time limit days-hrs:min:sec
#SBATCH --time {run_time}
# Exclude
# SBATCH --exclude=mind-1-26,mind-1-30
# Standard output and error log
#SBATCH --output={curr_dir}/slurm_out/{job_name}.out

conda activate brainiak_new
{script_name}
"""
    return sbatch_setup



#run fmri scripts python script
script_name = 'erd_fc'
script_name = 'gca_searchlight'

n =0 

for loc_sub,erd_sub in zip(sub_list['loc_sub'], sub_list['erd_sub']):
    

    for roi in rois:
        job_name = f'{erd_sub}_{roi}_{script_name}'
        script_path = f'python fmri/{script_name}.py {loc_sub} {erd_sub} {roi}'
        print(job_name)

        #create sbatch script
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(job_name, script_path))
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh")

        n+=1

     
        if n >= pause_crit:
            #wait X minutes
            time.sleep(pause_time*60)
            n = 0 
        

        