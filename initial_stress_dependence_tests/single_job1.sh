#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=10000

PARAMFILE=param1.txt
SAVEFILE=stress_dependence2

source /clusternfs/jrenton/anaconda2/my_anaconda.sh
python run_stress_dep.py 10000 -m 5 -d $SAVEFILE -P $PARAMFILE
