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
import pandas as pd

threshold_area_fraction = float(sys.argv[1])
death_to_birth_rate_ratio =  float(sys.argv[2])
domain_size_multiplier = float(sys.argv[3])
b = float(sys.argv[4])

NUMBER_SIMS = 1000
DELTA = 0.025
L = 10 # population size N=l*l
TIMEND = 200. # simulation time (hours)
MAX_POP_SIZE = 1000
TIMESTEP = 96. # time intervals to save simulation history
DEATH_RATE = 0.25/24.
INIT_TIME = 96.

PARENTDIR = "death_div_data/"
if not os.path.exists(PARENTDIR): # if the outdir doesn't exist create it
     os.makedirs(PARENTDIR)

game_constants = (b,1.)
game = lib.prisoners_dilemma_averaged
simulation = lib.simulation_contact_inhibition_area_dependent_event_data
init_simulation = lib.simulation_contact_inhibition_area_dependent
rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio)

with open(PARENTDIR+'info',"w") as f:
    f.write('death_rate = %.6f\n'%DEATH_RATE)
    f.write('initial pop size = %3d\n'%(L*L))
    f.write('domain width = %.1f\n'%(L*domain_size_multiplier))
    f.write('quiescent area ratio = %.1f'%threshold_area_fraction)
    f.write('death to birth rate ratio = %.2f'%death_to_birth_rate_ratio)
    f.write('timestep = %.1f'%TIMESTEP)


def run_single_unpack(args):
    return run_single(*args)

def run_single(i):
    """run a single voronoi tessellation model simulation"""
    sys.stdout.flush() 
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                init_time=INIT_TIME,til_fix=True,save_areas=True,init_simulation=init_simulation,
                return_events=True,save_cell_histories=False,N_limit=MAX_POP_SIZE,DELTA=DELTA,game=game,game_constants=game_constants,mutant_num=1,
                domain_size_multiplier=domain_size_multiplier,rates=rates,threshold_area_fraction=threshold_area_fraction)
    df = pd.DataFrame(history)
    df = df.assign(run=i)
    return df
    
def run_parallel():
    pool = Pool(cpu_count()-1,maxtasksperchild=1000)
    df = pd.concat([df for df in pool.imap(run_single,range(NUMBER_SIMS))])
    df.to_csv(PARENTDIR+"/alpha%.1f_db%.2f_b%.1f"%(threshold_area_fraction,death_to_birth_rate_ratio,b),index=False)
    pool.close()
    pool.join()
    
run_parallel()  
