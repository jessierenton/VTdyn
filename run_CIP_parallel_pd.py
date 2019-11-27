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

threshold_area_fraction = float(sys.argv[1])
death_to_birth_rate_ratio =  float(sys.argv[2])
domain_size_multiplier = float(sys.argv[3])
b = float(sys.argv[4])

NUMBER_SIMS = 10000
DELTA = 0.025
L = 10 # population size N=l*l
TIMEND = 80000. # simulation time (hours)
MAX_POP_SIZE = 1000
TIMESTEP = 96. # time intervals to save simulation history
DEATH_RATE = 0.25/24.
INIT_TIME = 96.

PARENTDIR = "CIP_pd_fix_N100/db%.2f_a%.1f/"%(death_to_birth_rate_ratio,threshold_area_fraction)
if not os.path.exists(PARENTDIR): # if the outdir doesn't exist create it
     os.makedirs(PARENTDIR)

game_constants = (b,1.)
game = lib.prisoners_dilemma_averaged
simulation = lib.simulation_contact_inhibition_area_dependent
rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio)

with open(PARENTDIR+'info',"w") as f:
    f.write('death_rate = %.3f\n'%DEATH_RATE)
    f.write('initial pop size = %3d\n'%(L*L))
    f.write('domain width = %3.1g\n'%(L*L*domain_size_multiplier))
    f.write('quiescent area ratio = %.1f'%threshold_area_fraction)
    f.write('death to birth rate ratio = %.2f'%death_to_birth_rate_ratio)
    f.write('timestep = %.1f'%TIMESTEP)

def fixed(history,i):
    if 0 not in history[-1].properties['type']:
        fix = 1  
    elif 1 not in history[-1].properties['type']:
        fix = 0
    else: 
        fix = -1
        data.save_N_mutant(history,PARENTDIR+'/incomplete_b%.1f'%b,i)
    return fix

def run_single_unpack(args):
    return run_single(*args)

def run_single(i):
    """run a single voronoi tessellation model simulation"""
    if i%100==0: sys.stdout.write(str(i)+'    ')
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                init_time=INIT_TIME,til_fix=True,save_areas=True,
                return_events=False,save_cell_histories=False,N_limit=MAX_POP_SIZE,DELTA=DELTA,game=game,game_constants=game_constants,mutant_num=1,
                domain_size_multiplier=domain_size_multiplier,rates=rates,threshold_area_fraction=threshold_area_fraction)
    return fixed(history,i)
    
def run_parallel():
    pool = Pool(cpu_count()-1,maxtasksperchild=1000)
    fixation = np.array([f for f in pool.imap(run_single,range(NUMBER_SIMS))]) 
    with open(PARENTDIR+'b%.1f'%b,'w') as wfile:    
        fixed = len(np.where(fixation==1)[0])
        lost = len(np.where(fixation==0)[0])
        incomplete = len(np.where(fixation==-1)[0])
        wfile.write('%d    %d    %d\n'%(fixed,lost,incomplete))


run_parallel()  
