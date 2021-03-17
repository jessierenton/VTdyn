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
import igraph
import pandas as pd

L = 10 # population size N=l*l
TIMEND = 7200. # simulation time (hours)
MAX_POP_SIZE = 1000
TIMESTEP = 96. # time intervals to save simulation history
DEATH_RATE = 0.25/24.
INIT_TIME = 96.

S0 = 1.

PARENTDIR = "CIP_proliferator_stats"
if not os.path.exists(PARENTDIR): # if the outdir doesn't exist create it
     os.makedirs(PARENTDIR)

with open(PARENTDIR+'/info',"w") as f:
    f.write('death_rate = %.3f\n'%DEATH_RATE)
    f.write('initial pop size = %3d\n'%(L*L))
    f.write('timestep = %.1f'%TIMESTEP)
simulation = lib.simulation_contact_inhibition_area_dependent  #simulation routine imported from lib
def number_proliferating(tissue,alpha):
    Zp = sum(tissue.mesh.areas>=np.sqrt(3)/2*alpha)
    return Zp

def get_number_proliferating_neighbours(tissue,neighbours,alpha):
    return sum(tissue.mesh.areas[neighbours] >= np.sqrt(3)/2*alpha)

def number_proliferating_neighbours(tissue,alpha):
    proliferating = np.where(tissue.mesh.areas>=np.sqrt(3)/2*alpha)[0]
    if len(proliferating) == 0:
        return np.array([0]) 
    return np.array([get_number_proliferating_neighbours(tissue,tissue.mesh.neighbours[i],alpha) for i in proliferating])
    
def create_pgraph(tissue,alpha):
    proliferating = np.where(tissue.mesh.areas>=np.sqrt(3)/2*alpha)[0]
    edges = list(set([tuple(sorted([i,np.where(proliferating==neighbour)[0][0]] ))
                for i,cell_id in enumerate(proliferating) 
                    for neighbour in tissue.mesh.neighbours[cell_id]
                        if neighbour in proliferating] )  )           
    return igraph.Graph(n=len(proliferating),edges=edges)

def number_clusters(history,alpha):
    return [len(create_pgraph(tissue,alpha).clusters()) for tissue in history]

def number_proliferating_neighbours_distribution(history,alpha,db):
    data = [np.bincount(number_proliferating_neighbours(tissue,alpha)) for tissue in history]
    Zp = [number_proliferating(tissue,alpha) for tissue in history]
    maxlen = max(len(nnp) for nnp in data)
    data = [np.pad(nnp,(0,maxlen-len(nnp)),'constant') for nnp in data]
    df = pd.DataFrame([{'np_{:d}'.format(i):f for i,f in enumerate(nnp)} for nnp in data])
    df.insert(0,'Zp',Zp)
    df.insert(1,'clusters',number_clusters(history,alpha))
    df.insert(0,'alpha',alpha)
    df.insert(0,'db',db)
    return df

def run_single_unpack(args):
    return run_single(*args)

def run_single(i,threshold_area_fraction,death_to_birth_rate_ratio,domain_size_multiplier,return_history=False):
    """run a single voronoi tessellation model simulation"""
    rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio)
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                init_time=INIT_TIME,til_fix=False,save_areas=True,
                return_events=False,save_cell_histories=False,N_limit=MAX_POP_SIZE,
                domain_size_multiplier=domain_size_multiplier,rates=rates,threshold_area_fraction=threshold_area_fraction)
    return number_proliferating_neighbours_distribution(history,threshold_area_fraction,death_to_birth_rate_ratio)
    
def run_parallel(paramfile,repeats):
    pool = Pool(cpu_count()-1,maxtasksperchild=1000)
    parameters = np.loadtxt(paramfile)
    args = [(i,threshold,death_to_birth_rate_ratio,domain_size_multiplier) 
                        for threshold,death_to_birth_rate_ratio,domain_size_multiplier in parameters
                        for i in range(repeats)]
    return pd.concat(pool.map(run_single_unpack,args))

paramfile = sys.argv[1]  
repeats = int(sys.argv[2])  
data = run_parallel(paramfile,repeats) 
data.to_csv(PARENTDIR+'/data',index=False)        
