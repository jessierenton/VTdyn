#!/bin/bash
#SBATCH --cpus-per-task=60
#SBATCH --array=0-2
#SBATCH --nodes=1
PARAMFILE=param_file_names.txt
if [ $(wc -l < $PARAMFILE) -ne $(($SLURM_ARRAY_TASK_MAX+1)) ]; then
	printf  "incorrect number of jobs" 1>&2
	exit 1
fi 
number=$(($SLURM_ARRAY_TASK_ID+1))
param_file=$(sed -n "$number"p $PARAMFILE)
source /clusternfs/jrenton/anaconda2/my_anaconda.sh
python run_stress_dep.py 500 -d stress_data -m 10 -P $param_file
