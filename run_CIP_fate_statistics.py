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
TIMEND = 2400. # simulation time (hours)
MAX_POP_SIZE = 1000
TIMESTEP = 96. # time intervals to save simulation history
DEATH_RATE = 0.25/24.
INIT_TIME = 96.

S0 = 1.

PARENTDIR = "CIP_fate_statistics"

with open(PARENTDIR+'/info',"w") as f:
    f.write('death_rate = %.3f\n'%DEATH_RATE)
    f.write('initial pop size = %3d\n'%(L*L))
    f.write('domain width = %.1f\n'%(L*S0))
    f.write('timestep = %.1f'%TIMESTEP)
simulation = lib.simulation_contact_inhibition_area_dependent  #simulation routine imported from lib

def run_single_unpack(args):
    return run_single(*args)

def run_single(i,threshold_area_fraction,death_to_birth_rate_ratio,domain_size_multiplier,return_history=False):
    """run a single voronoi tessellation model simulation"""
    rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio)
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                init_time=INIT_TIME,til_fix=False,save_areas=True,
                return_events=False,save_cell_histories=True,N_limit=MAX_POP_SIZE,
                domain_size_multiplier=domain_size_multiplier,rates=rates,threshold_area_fraction=threshold_area_fraction)
    outdir = PARENTDIR+'/db%.1f_a%.2f_$d'%(death_to_birth_rate_ratio,threshold_area_fraction,i)
    data.save_as_json(history,outdir,['cell_histories'],parameters,index=0)
    
def run_parallel(paramfile,repeats):
    pool = Pool(cpu_count()-1,maxtasksperchild=1000)
    parameters = np.loadtxt(paramfile)
    args = [(i,threshold,death_to_birth_rate_ratio,domain_size_multiplier) 
                        for threshold,death_to_birth_rate_ratio,domain_size_multiplier in parameters
                        for i in range(repeats)]
    pool.map(run_single_unpack,args) 

paramfile = sys.argv[1]  
repeats = int(sys.argv[2])  
run_parallel(paramfile,repeats)          
