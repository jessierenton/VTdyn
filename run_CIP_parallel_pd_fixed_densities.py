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
area_ratio = round(domain_size_multiplier ** 2, 2)
b = float(sys.argv[4])
job_id = sys.argv[5]

NUMBER_SIMS = 1
BATCH_SIZE = 1000
DELTA = 0.025
L = 10 # population size N=l*l
TIMEND = 80. # simulation time (hours)
MAX_POP_SIZE = 1000
TIMESTEP = 96. # time intervals to save simulation history
DEATH_RATE = 0.25/24.
INIT_TIME = 96.

PARENTDIR = "CIP_pd_fix_N100_fixed_density/db%.2f_ar%.1f/"%(death_to_birth_rate_ratio,area_ratio)
if not os.path.exists(PARENTDIR): # if the outdir doesn't exist create it
     os.makedirs(PARENTDIR)

game_constants = (b,1.)
game = lib.prisoners_dilemma_averaged
simulation = lib.simulation_contact_inhibition_area_dependent
rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio)

with open(PARENTDIR+'info',"w") as f:
    f.write('death_rate = %.6f\n'%DEATH_RATE)
    f.write('initial pop size = %3d\n'%(L*L))
    f.write('domain width = %.1f\n'%(L*domain_size_multiplier))
    f.write('area ratio = %.2f'%area_ratio)
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
    sys.stdout.flush() 
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                init_time=INIT_TIME,til_fix=True,save_areas=True,
                return_events=False,save_cell_histories=False,N_limit=MAX_POP_SIZE,DELTA=DELTA,game=game,game_constants=game_constants,mutant_num=1,
                domain_size_multiplier=domain_size_multiplier,rates=rates,threshold_area_fraction=threshold_area_fraction)
    fixation = fixed(history,i)
    mean_pop_size = np.mean([len(tissue) for tissue in history])
    with open(PARENTDIR+'b%.2f_%s_time'%(b,job_id),'a') as wfile:
        wfile.write('%5d    %5d    %d\n'%(i,history[-1].time,fixation))
    return fixation, mean_pop_size
    
def run_parallel():
    pool = Pool(cpu_count(),maxtasksperchild=1000)
    results = [result for result in pool.imap(run_single,range(NUMBER_SIMS))]
    fixation = np.array([result[0] for result in results])
    Nmean = np.mean([result[1] for result in results])
    with open(PARENTDIR+'b%.2f_%s'%(b,job_id),'w') as wfile:    
        if NUMBER_SIMS%BATCH_SIZE != 0: 
            batch_size=1
        else: 
            batch_size = BATCH_SIZE
        fixation = fixation.reshape((NUMBER_SIMS/batch_size,batch_size))
        for fixation_batch in fixation:
            fixed = len(np.where(fixation_batch==1)[0])
            lost = len(np.where(fixation_batch==0)[0])
            incomplete = len(np.where(fixation_batch==-1)[0])
            wfile.write('%d    %d    %d    %d\n'%(fixed,lost,incomplete, Nmean))

run_parallel()  