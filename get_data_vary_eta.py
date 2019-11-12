from multiprocessing import Pool
import numpy as np
import libs.pd_lib_neutral as lib #library for simulation routines
import libs.data as data
# import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import pandas as pd
import os

"""run a single voronoi tessellation model simulation"""

outdir = 'VTcoupled_force_area_data_vary_eta'
dt = 0.005
if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)

def run_sim(eta):
    print(eta)
    rand = np.random.RandomState()
    history = lib.run_simulation_vary_eta(simulation,l,timestep,timend,rand,eta,dt,til_fix=False,save_areas=True)
    return (data.mean_force(history,False),data.mean_area(history,False))

l = 10 # population size N=l*l
timend = 1000 # simulation time (hours)
timestep = 1. # time intervals to save simulation history

rand = np.random.RandomState()
simulation = lib.simulation_ancestor_tracking  #simulation routine imported from lib

pool = Pool(maxtasksperchild=1000) # creating a pool of workers to run simulations in parallel

eta_vals = np.array((0.7,0.8,0.9,0.95,1.0,2.0,5.0,10.0,50.0,500.),dtype=float)
hdata = {"eta":np.repeat(eta_vals,int(timend/timestep+1)),"time":np.tile(np.linspace(0,timend,timend/timestep+1),len(eta_vals))}
fa_data = np.array([fa_d for fa_d in pool.imap(run_sim,eta_vals)])

hdata["force"] = (fa_data[:,0,:]).flatten()
hdata["area"] = (fa_data[:,1,:]).flatten()

df = pd.DataFrame(data=hdata)

df.to_csv(outdir+"/data")