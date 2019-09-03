#!/bin/bash
#SBATCH --cpus-per-task=60
#SBATCH --array=0-3
#SBATCH --nodes=1

source /clusternfs/jrenton/anaconda2/my_anaconda.sh
python run_CIP_parallel.py param${SLURM_ARRAY_TASK_ID}.txt 3
