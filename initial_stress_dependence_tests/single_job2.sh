#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=15000

PARAMFILE=param2.txt
SAVEFILE=stress_dependence2

source /clusternfs/jrenton/anaconda2/my_anaconda.sh
python run_stress_dep.py 10000 -m 5 -d $SAVEFILE -P $PARAMFILE
