#!/bin/bash
#SBATCH --cpus-per-task=60
#SBATCH --array=0-7
#SBATCH --nodes=1
PARAMFILE=pd_params9/p0
number=$(($SLURM_ARRAY_TASK_ID+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`
p2=`(sed -n "$number"p $PARAMFILE) | awk '{print $2}'`
p3=`(sed -n "$number"p $PARAMFILE) | awk '{print $3}'`
p4=`(sed -n "$number"p $PARAMFILE) | awk '{print $4}'`

source /clusternfs/jrenton/anaconda2/my_anaconda.sh

python run_CIP_parallel_pd.py $p1 $p2 $p3 $p4 ${SLURM_ARRAY_JOB_ID}
