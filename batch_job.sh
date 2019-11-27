#!/bin/bash
#SBATCH --cpus-per-task=80
#SBATCH --array=0-2
#SBATCH --nodes=1

source /clusternfs/jrenton/anaconda2/my_anaconda.sh
python run_CIP_parallel.py ${SLURM_ARRAY_TASK_ID}
