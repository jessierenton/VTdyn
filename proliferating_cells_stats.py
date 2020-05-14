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

def mean_proportion_coop_neighbours(idxlist,all_neighbours,cooperators):
    neighbours = [all_neighbours[i] for i in idxlist]
    return np.mean([float(sum(cooperators[cell_neighbours]))/len(cell_neighbours) for cell_neighbours in neighbours])

def number_proliferating_cells(tissue,alpha):
    cooperators = np.array(tissue.properties['type'],dtype=bool)
    proliferating = tissue.mesh.areas > alpha*A0
    proliferating_cooperators = np.where(proliferating[cooperators])[0]
    proliferating_defectors = np.where(proliferating[~cooperators])[0]
    propccready = mean_proportion_coop_neighbours(proliferating_cooperators,tissue.mesh.neighbours,cooperators)
    propcdready = mean_proportion_coop_neighbours(proliferating_defectors,tissue.mesh.neighbours,cooperators)
    return sum(cooperators),len(tissue),sum(proliferating[cooperators]),sum(proliferating),propccready,propcdready

def run_sim(alpha,db,m,DELTA,game,game_constants,i):
    """run a single simulation and save interaction data for each clone"""
    rates = (DEATH_RATE,DEATH_RATE/db)
    rand = np.random.RandomState()
    data = [number_proliferating_cells(tissue,alpha)
                for tissue in lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                                    init_time=INIT_TIME,til_fix='exclude_final',save_areas=True,return_events=False,save_cell_histories=False,
                                    N_limit=MAX_POP_SIZE,DELTA=DELTA,game=game,game_constants=game_constants,
                                    mutant_num=1,mutant_type=mutant_type,domain_size_multiplier=m,rates=rates,threshold_area_fraction=alpha,generator=True)]                
    if i%100 == 0:
        print('%d complete'%i)
    return data

def sort_data(data):
    data = [[i,n,N,proliferating_coop,proliferating_all,propccready,propcdready]
                for i,run_data in enumerate(data)
                    for n,N,proliferating_coop,proliferating_all,propccready,propcdready in run_data]
    df = pd.DataFrame(data,columns = ['run','n','N','n_ready','N_ready','propccready','propcdready'])
    return df

L = 10 # population size N = l*l
INIT_TIME = 96. # initial simulation time to equilibrate 
TIMEND = 80. # length of simulation (hours)
TIMESTEP = 12. # time intervals to save simulation history
MAX_POP_SIZE = 1000
DEATH_RATE = 0.25/24.
SIM_RUNS = int(sys.argv[1]) # number of sims to run
n_min = 1 
simulation = lib.simulation_contact_inhibition_area_dependent 

b = sys.argv[2]
if b == 'None':
    b = None
    DELTA,game,game_constants = 0.,None,None
else:
    b = float(b)   
    DELTA,game,game_constants = 0.025,lib.prisoners_dilemma_averaged,(b,1)


params = [[0.800000, 0.100000, 0.859628],[1.000000, 0.100000, 0.948836],[1.200000, 0.100000, 1.030535]]
alpha,db,m = params[int(sys.argv[3])]

mutant_type=int(sys.argv[4])

outdir = 'CD_data/cip/'

if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)

# run simulations in parallel 
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
df = sort_data(pool.map(partial(run_sim,alpha,db,m,DELTA,game,game_constants),range(SIM_RUNS)))
pool.close()
pool.join()
if b is None:
    df.to_csv(outdir+'db%.2f_alpha%.1f_neutral.csv'%(db,alpha),index=False)
else:
    df.to_csv(outdir+'db%.2f_alpha%.1f_b%.2f_mutant%d.csv'%(db,alpha,b,mutant_type),index=False)