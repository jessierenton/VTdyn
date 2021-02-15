from multiprocessing import Pool  #parallel processing
import multiprocessing as mp
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth
import structure.initialisation as init
import sys,os
import numpy as np
import libs.contact_inhibition_lib as lib
import libs.data as data
from functools import partial
import pandas as pd

def mean_proportion_coop_neighbours_ready(idxlist,all_neighbours,cooperators):
    neighbours = [all_neighbours[i] for i in idxlist]
    data = [float(sum(cooperators[cell_neighbours]))/len(cell_neighbours) for cell_neighbours in neighbours]
    return np.mean(data),np.std(data)
    
def proliferating_cells_data(tissue,alpha,mutant_index,n):
    cooperators = tissue.properties['ancestor']==mutant_index
    proliferating = tissue.mesh.areas > alpha*A0
    proliferating_cooperators = np.where(proliferating&cooperators)[0]
    proliferating_defectors = np.where(proliferating&~cooperators)[0]
    prop_neighbours_coop,prop_neighbours_coop_std = mean_proportion_coop_neighbours(np.where(cooperators)[0],tissue.mesh.neighbours,cooperators)
    prop_neighbours_defect,prop_neighbours_defect_std = mean_proportion_coop_neighbours(np.where(~cooperators)[0],tissue.mesh.neighbours,cooperators)
    prop_neighbours_coop_ready,prop_neighbours_coop_ready_std = mean_proportion_coop_neighbours(proliferating_cooperators,tissue.mesh.neighbours,cooperators)
    prop_neighbours_defect_ready,prop_neighbours_defect_ready_std = mean_proportion_coop_neighbours(proliferating_defectors,tissue.mesh.neighbours,cooperators)
    return (n,len(tissue)-n,len(proliferating_cooperators),len(proliferating_defectors),
                    prop_neighbours_coop,prop_neighbours_coop_std,prop_neighbours_defect,prop_neighbours_defect_std,
                        prop_neighbours_coop_ready,prop_neighbours_coop_ready_std,prop_neighbours_defect_ready,prop_neighbours_defect_ready_std)
        
        
        

def run_sim(alpha,db,m,i):
    """run a single simulation and save interaction data for each clone"""
    rates = (DEATH_RATE,DEATH_RATE/db)
    rand = np.random.RandomState()
    data = [(tissue.time,proliferating_cells_data(tissue,alpha,mutant_index,n))
                for tissue in lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,ancestors=True,
                                    init_time=INIT_TIME,til_fix=True,save_areas=True,return_events=False,save_cell_histories=False,
                                    N_limit=MAX_POP_SIZE,game=None, mutant_num=None,domain_size_multiplier=m,rates=rates,threshold_area_fraction=alpha,generator=True)
                                for mutant_index,n in enumerate(np.bincount(tissue.properties['ancestor'])) if n>=n_min]                
    if i%100 == 0:
        print('%d complete'%i)
    return data

def sort_data(data):
    data = [[i,round(time),nc,nd,proliferating_coop,proliferating_defect,
                prop_neighbours_coop,prop_neighbours_coop_std,prop_neighbours_defect,prop_neighbours_defect_std,
                prop_neighbours_coop_ready,prop_neighbours_coop_ready_std,prop_neighbours_defect_ready,prop_neighbours_defect_ready_std]
                for i,run_data in enumerate(data)
                    for time,(nc,nd,proliferating_coop,proliferating_defect,
                    prop_neighbours_coop,prop_neighbours_coop_std,prop_neighbours_defect,prop_neighbours_defect_std,
                    prop_neighbours_coop_ready,prop_neighbours_coop_ready_std,prop_neighbours_defect_ready,prop_neighbours_defect_ready_std
                    ) in run_data]
    df = pd.DataFrame(data,columns = ['run','time','nc','nd','ncready','ndready',
                                        'propcc','propccstd','propcd','propcdstd',
                                        'propccready','propccreadystd','propcdready','propcdreadystd'])
    return df

L = 10 # population size N = l*l
INIT_TIME = 96. # initial simulation time to equilibrate 
TIMEND = 80000. # length of simulation (hours)
TIMESTEP = 96. # time intervals to save simulation history
MAX_POP_SIZE = 1000
DEATH_RATE = 0.25/24.
SIM_RUNS = int(sys.argv[1]) # number of sims to run
n_min = 1 
simulation = lib.simulation_contact_inhibition_area_dependent 

params = [[0.800000, 0.100000, 0.859628],[1.000000, 0.100000, 0.948836],[1.200000, 0.100000, 1.030535]]
alpha,db,m = params[int(sys.argv[2])]

outdir = 'CD_data/cip/'

if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)

# run simulations in parallel 
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
df = sort_data(pool.map(partial(run_sim,alpha,db,m),range(SIM_RUNS)))
pool.close()
pool.join()
df.to_csv(outdir+'db%.2f_alpha%.1f_neutral.csv'%(db,alpha),index=False)
