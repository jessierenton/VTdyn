#!/bin/bash
#$ -pe smp 48
#$ -l exclusive
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -o jobs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -t 1-54
module load anaconda2
source activate VTenv

INPUT_ARGS=$(sed -n "$SGE_TASK_ID"p params_simple1.txt)
python run_CIP_parallel_pd.py $INPUT_ARGS ${$SGE_TASK_ID}