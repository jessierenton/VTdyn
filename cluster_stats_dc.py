from multiprocessing import Pool  #parallel processing
import multiprocessing as mp
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth
import structure.initialisation as init
import sys
import os
import numpy as np
import libs.contact_inhibition_lib as lib
import libs.data as data
from functools import partial

def calc_interactions(tissue,mutant_index,n):
    """treats all cells with ancestor 'mutant_index' as cooperators
    returns:
        n (int): size of clone
        I_CC/I_CD (ints): number of cooperator-cooperator/defector interactions in population
        W_CC/W_CD (floats): number of cooperator-cooperator/defector interactions in pop. weighted by neighbour number    
    """
    N = len(tissue)
    neighbours = tissue.mesh.neighbours
    types = tissue.properties['ancestor']==mutant_index
    I_CC,I_CD,W_CC,W_CD,N_D = 0,0,0.,0.,0
    for ctype,cell_neighbours in zip(types,neighbours):
        if ctype:
            Cneigh,neigh = float(sum(types[cell_neighbours])),float(len(cell_neighbours))
            I_CC += Cneigh
            I_CD += neigh - Cneigh
            W_CC += Cneigh/neigh
            W_CD += (neigh-Cneigh)/neigh
    return [n,N,I_CC,I_CD,W_CC,W_CD]

def run_sim(m,i):
    """run a single simulation and save interaction data for each clone"""
    rates = (DEATH_RATE,)
    rand = np.random.RandomState()
    data = [calc_interactions(tissue,mutant_index,n)
                for tissue in lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                                    init_time=INIT_TIME,til_fix=True,save_areas=False,return_events=False,save_cell_histories=False,
                                    N_limit=MAX_POP_SIZE,game=None,mutant_num=None,domain_size_multiplier=m,rates=rates,generator=True)
                for mutant_index,n in enumerate(np.bincount(tissue.properties['ancestor'])) if n>=n_min]                
    outdir1 = outdir + 'm%.3f/'%m
    if not os.path.exists(outdir1): # if the outdir doesn't exist create it
         os.makedirs(outdir1)
    np.savetxt('%sdata_%d'%(outdir1,i),data,fmt=('%4d','%4d','%4d','%4d','%4.6f','%4.6f'))
    return None

L = 10 # population size N = l*l
INIT_TIME = 96. # initial simulation time to equilibrate 
TIMEND = 80000. # length of simulation (hours)
TIMESTEP = 96. # time intervals to save simulation history
MAX_POP_SIZE = 1000
DEATH_RATE = 0.25/24.
SIM_RUNS = int(sys.argv[1]) # number of sims to run taken as command line arg
n_min = 2 
simulation = lib.simulation_decoupled_update
outdir = 'interaction_data_dc/raw_data/'
if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)

domain_size_multipliers = [0.85,0.9,0.95,1.0,1.05,1.1,1.15]



# run simulations in parallel 
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
for m in domain_size_multipliers:
    map(partial(run_sim,m),range(SIM_RUNS))
pool.close()
pool.join()
