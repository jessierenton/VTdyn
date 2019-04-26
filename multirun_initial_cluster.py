from multiprocessing import Process,Pool,Lock  #parallel processing
import multiprocessing as mp
from itertools import repeat
import sys
from functools import partial

import os
import numpy as np
import libs.pd_lib as lib
import libs.data as data
# import libs.plot as vplt
from libs.pd_lib import run,simulation_decoupled_update,prisoners_dilemma_averaged,simulation_initialise_tissue_with_cluster
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth, MutantSpringForce
import structure.initialisation as init


game = prisoners_dilemma_averaged
b = float(sys.argv[1])
sim_runs=int(sys.argv[2])
cluster_sizes = np.array(sys.argv[3:],dtype=int)

#prisoner dilemma params
c,DELTA = 1.0,0.025

rand = np.random.RandomState()
outdir = 'pd_av_clusters_b_%d'%(b)

l = 10
timend = 10000.
timestep = 12.0



if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)
     
info = """
donation game poisson process fixed N

T_D = 12.
L0 = 1.0
EPS = 0.05

# Osborne 2017 params
MU = -50.
ETA = 1.0
dt = 0.005 #hours

maxtime = %.0f
timestep = %.0f

N = %d

c,DELTA = %.1f, %.3f
""" %(timend,timestep,l*l,c,DELTA)

with open(outdir+'/info',"w",0) as infofile:
    infofile.write(info)

def run_simulation_cluster(l,timestep,timend,rand,DELTA,game,game_constants,cluster_size,save_areas=False):
    """prisoners dilemma with decoupled birth and death"""
    tissue = init.init_tissue_torus(l,l,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue = lib.simulation_initialise_tissue_with_cluster(tissue,dt,10/dt,rand,cluster_size,main_type=0)
    history = run(tissue, simulation_decoupled_update(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,game_constants),timend/dt,timestep/dt)
    return history

def run_parallel(b,cluster_size,i):
    rand=np.random.RandomState()
    history = run_simulation_cluster(l,timestep,timend,rand,DELTA,prisoners_dilemma_averaged,(b,c),cluster_size,save_areas=False)
    if 0 not in history[-1].properties['type']:
        fix = 1  
        # data.save_N_mutant(history,outdir+'/fixed_b%.1f'%b,i)
    elif 1 not in history[-1].properties['type']:
        fix = 0
    else: 
        fix = -1
        data.save_N_mutant(history,outdir+'/incomplete_b%.1f'%b,i)
    return fix

cpunum=mp.cpu_count()
pool = Pool(processes=cpunum,maxtasksperchild=1000) # creating a pool with processors equal to the number of processors
fix_results = open(outdir+'/fix_b%d'%b,'w',0)
fix_results.write('#initial_size | fixed  | lost |  incomplete\n')
for cluster_size in cluster_sizes:
    text = '\r running cluster size %d'%(cluster_size)
    sys.stdout.write(text)
    sys.stdout.flush()
    fixation = np.array([f for f in pool.imap(partial(run_parallel,b,cluster_size),range(sim_runs))])  # mapping of all the calls necessary into the calling function
    fixed = len(np.where(fixation==1)[0])
    nofixed = len(np.where(fixation==0)[0])
    incomplete = len(np.where(fixation==-1)[0])
    fix_results.write('       %2d        %3d    %3d     %3d\n'%(cluster_size,fixed,nofixed,incomplete))
