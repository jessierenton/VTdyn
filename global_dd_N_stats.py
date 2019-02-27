import numpy as np
import libs.pd_lib_density_t_decouple as lib #library for simulation routines
import libs.data as data
# import libs.plot as vplt #plotting library
import os
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
from multiprocessing import Pool #parallel processing

"""run a single voronoi tessellation model simulation"""

outdir = 'global_dd_death_N_data'
if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)
l = 10 # population size N=l*l
timend = 2000. # simulation time (hours)
timestep = 1. # time intervals to save simulation history

b,c,DELTA = 3.,1.0,0.0 #prisoner's dilemma game parameters
KAPPA_vals = (0.1,1.0,10.,100.,1000.)
repeats = 2

simulation = lib.simulation_pd_global_density_dep  #simulation routine imported from lib
game = lib.prisoners_dilemma_averaged #game imported from lib
game_parameters = (b,c)
params = {'KAPPA':,'N0':l*l}

def run_sim(KAPPA):
    params['KAPPA']=KAPPA
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,l,timestep,timend,rand,params,DELTA,game,game_parameters,til_fix=False,mutant_num=1)
    return [len(tissue) for tissue in history]
    
pool = Pool(maxtasksperchild=1000)
KAPPA_repeats = np.repeat(KAPPA_vals,repeats)
for i,N_vals in enumerate(pool.imap(run_sim,KAPPA_repeats)):
    np.savetxt(outdir+'/K_%.1f_%d'%(KAPPA_repeats[i],i%repeats),N_vals)
