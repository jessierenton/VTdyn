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

def get_areas_and_fitnesses(tissue,DELTA,game,game_constants):
    areas = tissue.mesh.areas
    cooperators = np.array(tissue.properties['type'],dtype=bool)
    get_fitnesses(tissue,DELTA,game,game_constants,cooperators)
    return sum(cooperators),len(tissue),get_areas(areas,cooperators),get_fitnesses(tissue,DELTA,game,game_constants,cooperators)
    
def get_areas(areas,cooperators):
    cooperator_areas,defector_areas = areas[cooperators],areas[~cooperators]
    return np.mean(cooperator_areas),np.std(cooperator_areas),np.mean(defector_areas),np.std(defector_areas)

def get_fitnesses(tissue,DELTA,game,game_constants,cooperators):
    fitnesses = lib.recalculate_fitnesses(tissue.mesh.neighbours,tissue.properties['type'],DELTA,game,game_constants)
    cooperator_fitnesses,defector_fitnesses = fitnesses[cooperators],fitnesses[~cooperators]
    return np.mean(cooperator_fitnesses),np.std(cooperator_fitnesses),np.mean(defector_fitnesses),np.std(defector_fitnesses)

def run_sim(alpha,db,m,DELTA,game,game_constants,i):
    """run a single simulation and save interaction data for each clone"""
    rates = (DEATH_RATE,DEATH_RATE/db)
    rand = np.random.RandomState()
    data = [get_areas_and_fitnesses(tissue,DELTA,game,game_constants)
                for tissue in lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                                    init_time=INIT_TIME,til_fix='exclude_final',save_areas=True,return_events=False,save_cell_histories=False,
                                    N_limit=MAX_POP_SIZE,DELTA=DELTA,game=game,game_constants=game_constants,
                                    mutant_num=1,domain_size_multiplier=m,rates=rates,threshold_area_fraction=alpha,generator=True)]                

    return data

def sort_data(data):
    data = [[n,N,area_data,fitness_data ]
                for run_data in data
                    for n,N,area_data,fitness_data in run_data]
    area_data = [[n,N,coop_mean,coop_std,defect_mean,defect_std] 
                    for n,N,(coop_mean,coop_std,defect_mean,defect_std),fitness_data in data]
    area_data = [[n,N,coop_mean,coop_std,defect_mean,defect_std] 
                    for n,N,(coop_mean,coop_std,defect_mean,defect_std),fitness_data in data]
    area_df = pd.DataFrame(area_data,columns = ['n','N','coop_area','coop_sd','defect_area','defect_sd'])
    fitness_data = [[n,N,coop_mean,coop_std,defect_mean,defect_std] 
                    for n,N,area_data,(coop_mean,coop_std,defect_mean,defect_std) in data]
    fitness_df = pd.DataFrame(fitness_data,columns = ['n','N','coop_fitness','coop_sd','defect_fitness','defect_sd'])
    return area_df,fitness_df

L = 10 # population size N = l*l
INIT_TIME = 96. # initial simulation time to equilibrate 
TIMEND = 80000. # length of simulation (hours)
TIMESTEP = 12. # time intervals to save simulation history
MAX_POP_SIZE = 1000
DEATH_RATE = 0.25/24.
SIM_RUNS = int(sys.argv[1]) # number of sims to run
n_min = 1 
simulation = lib.simulation_contact_inhibition_area_dependent     
DELTA,game,game_constants = 0.025,lib.prisoners_dilemma_averaged,(4.,1.)

params = [[0.800000, 0.100000, 0.859628],[1.000000, 0.100000, 0.948836],[1.200000, 0.100000, 1.030535]]
alpha,db,m = params[int(sys.argv[2])]

outdir = 'CD_data_b%.1f/'%game_constants[0]
outdir_area = outdir+'area/'
outdir_fitness = outdir+'fitness/'
if not os.path.exists(outdir_area): # if the outdir doesn't exist create it
     os.makedirs(outdir_area)
if not os.path.exists(outdir_fitness): # if the outdir doesn't exist create it
     os.makedirs(outdir_fitness)

# run simulations in parallel 
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
area_df,fitness_df = sort_data(pool.map(partial(run_sim,alpha,db,m,DELTA,game,game_constants),range(SIM_RUNS)))
pool.close()
pool.join()
area_df.to_csv(outdir_area+'db%.2f_alpha%.1f.csv'%(db,alpha),index=False)
fitness_df.to_csv(outdir_fitness+'db%.2f_alpha%.1f.csv'%(db,alpha),index=False)