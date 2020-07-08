from multiprocessing import Pool  #parallel processing
import multiprocessing as mp
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth
import structure.initialisation as init
import sys
import os
import numpy as np
import libs.pd_lib_neutral as lib
import libs.data as data
from functools import partial

def number_mutants(tissue,mutant_id):
    return sum(properties['ancestor']==mutant_id)

def number_coop_neighbours(tissue,types,cell):
    neighbour_types = types[tissue.mesh.neighbours[cell]]
    return sum(neighbour_types)

def freq_coop_neighbours(tissue,mutant_id):
    """returns [number of cooperator cells, freq distribution for number of cooperator neighbours (cooperators),
        freq distr. for number of cooperator neighbours (defectors)] """
    types = (tissue.properties['ancestor']==mutant_id)*1
    coop_neighbours_defect = np.array([number_coop_neighbours(tissue,types,cell) for cell,cell_type in enumerate(types) if cell_type==0],dtype=int)
    coop_neighbours_coop = np.array([number_coop_neighbours(tissue,types,cell) for cell,cell_type in enumerate(types) if cell_type==1],dtype=int)
    return sum(types),np.bincount(coop_neighbours_coop)/float(sum(types)),np.bincount(coop_neighbours_defect)/float(len(tissue)-sum(types))

def distribution_to_string(dist):
    return '    '.join(['%.5f'%freq for freq in dist])

def calc_coop_neighbours(history,outdir):
    mutant_id = history[-1].properties['ancestor'][0]  
    for tissue in history:
        number_coop,freq_coop_neighbours_coop,freq_coop_neighbours_defect = freq_coop_neighbours(tissue,mutant_id) 
        with open(outdir+'coop_%.3d'%number_coop,'a') as f:
            f.write(distribution_to_string(freq_coop_neighbours_coop))
            f.write('\n')
        with open(outdir+'defect_%.3d'%number_coop,'a') as f:
            f.write(distribution_to_string(freq_coop_neighbours_defect))
            f.write('\n')
            
        

def run_sim(m,i):
    """run a single simulation and save interaction data for each clone"""
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                                    init_time=INIT_TIME,til_fix=True,save_areas=False)  
                             
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
simulation = lib.simulation_ancestor_tracking
outdir = 'coop_neighbour_distribution/'
if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)



# run simulations in parallel 
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
for m in domain_size_multipliers:
    pool.map(partial(run_sim,m),range(SIM_RUNS))
pool.close()
pool.join()
