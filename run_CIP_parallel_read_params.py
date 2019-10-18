import numpy as np
# from pathos.multiprocessing import cpu_count
# from pathos.pools import ParallelPool as Pool
from multiprocessing import Pool,cpu_count
import libs.contact_inhibition_lib as lib #library for simulation routines
import libs.data as data
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import sys,os
import itertools 

L = 10 # population size N=l*l
TIMEND = 1000. # simulation time (hours)
MAX_POP_SIZE = 500
TIMESTEP = 10. # time intervals to save simulation history
DEATH_RATE = 1./12
INIT_TIME = None

# DATA_SAVE_FIELDS = ["pop_size","cell_histories","cycle_phases","density",
#                     "cell_seperation"]
DATA_SAVE_FIELDS = ["density","cell_histories","areas","pop_size"]

for d in DATA_SAVE_FIELDS:
    if d not in data.FIELDS_DICT:
        raise ValueError("not all data types are correct")

PARENTDIR = "CIP_data_area_threshold/sweep_fixed_N100"

with open(PARENTDIR+'/info',"w") as f:
    f.write('death_rate = %.3f\n'%DEATH_RATE)
    f.write('initial pop size = %3d'%(L*L))
    # f.write('domain width = %3.1g'%(L*L*S0) )
simulation = lib.simulation_contact_inhibition_area_dependent  #simulation routine imported from lib

def run_single_unpack(args):
    return run_single(*args)

def run_single(i,threshold_area_fraction,death_to_birth_rate_ratio,domain_size_multiplier):
    """run a single voronoi tessellation model simulation"""
    rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio)
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                init_time=INIT_TIME,til_fix=False,save_areas=True,
                return_events=False,save_cell_histories=True,N_limit=MAX_POP_SIZE,
                domain_size_multiplier=S0,rates=rates,threshold_area_fraction=threshold_area_fraction)
    outdir = PARENTDIR+'/DtoB%.1f_Thresh%.2f'%(death_to_birth_rate_ratio,threshold_area_fraction)
    if len(DATA_SAVE_FIELDS) > 0:
        data.save_as_json(history,outdir,DATA_SAVE_FIELDS,{"threshold_area_fraction":threshold_area_fraction,
                    "death_rate":DEATH_RATE,"death_to_birth_rate_ratio":death_to_birth_rate_ratio,
                    "width":history[0].mesh.geometry.width},i)
    
def run_parallel(paramfile,repeats):
    pool = Pool(cpu_count()-1,maxtasksperchild=1000)
    parameters = np.loadtxt(paramfile)
    args = [(i,threshold,death_rate,domain_size_multiplier) 
                        for threshold,death_rate,domain_size_multiplier in parameters
                        for i in range(repeats)]
    pool.map(run_single_unpack,args)

paramfile = sys.argv[1]  
repeats = int(sys.argv[2])  
run_parallel(paramfile,repeats)          
