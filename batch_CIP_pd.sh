#!/bin/bash
#SBATCH --cpus-per-task=80
#SBATCH --array=0-3
#SBATCH --nodes=1
PARAMFILE=pd_params0
if [ $(wc -l < $PARAMFILE) -ne $(($SLURM_ARRAY_TASK_MAX+1)) ]; then
        printf  "incorrect number of jobs" 1>&2
        exit 1
fi
number=$(($SLURM_ARRAY_TASK_ID+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`
p2=`(sed -n "$number"p $PARAMFILE) | awk '{print $2}'`
p3=`(sed -n "$number"p $PARAMFILE) | awk '{print $3}'`
p4=`(sed -n "$number"p $PARAMFILE) | awk '{print $4}'`

source /clusternfs/jrenton/anaconda2/my_anaconda.sh

python run_CIP_parallel_pd.py $p1 $p2 $p3 $p4
