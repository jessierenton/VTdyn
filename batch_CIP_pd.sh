#!/bin/bash
#SBATCH --cpus-per-task=88
#SBATCH --array=0-1
#SBATCH --nodes=1
PARAMFILE=pd_params8/p3
number=$(($SLURM_ARRAY_TASK_ID+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`
p2=`(sed -n "$number"p $PARAMFILE) | awk '{print $2}'`
p3=`(sed -n "$number"p $PARAMFILE) | awk '{print $3}'`
p4=`(sed -n "$number"p $PARAMFILE) | awk '{print $4}'`

source /clusternfs/jrenton/anaconda2/my_anaconda.sh

python run_CIP_parallel_pd.py $p1 $p2 $p3 $p4 ${SLURM_ARRAY_JOB_ID}
