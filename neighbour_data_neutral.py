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
import pandas as pd

def distribution_data(history,coop_id,i):
    return [{'tissueid':i,'time':int(tissue.time),'n':sum(tissue.properties['ancestor']==coop_id),'k':len(cell_neighbours),
                'j':sum((tissue.properties['ancestor']==coop_id)[cell_neighbours]),'type': 1 if tissue.properties['ancestor'][idx]==coop_id else 0} 
                for tissue in history if 1<=sum(tissue.properties['ancestor']==coop_id)<100
                    for idx,cell_neighbours in enumerate(tissue.mesh.neighbours)]     

def run_sim(i):
    """run a single simulation and save interaction data for each clone"""
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                                    init_time=INIT_TIME,til_fix='exclude_final',save_areas=False)  
    coop_id = np.argmax(np.bincount(history[-1].properties['ancestor']))
    return distribution_data(history,coop_id,i)

L = 10 # population size N = l*l
INIT_TIME = 96. # initial simulation time to equilibrate 
TIMEND = 80000. # length of simulation (hours)
TIMESTEP = 12. # time intervals to save simulation history
MAX_POP_SIZE = 1000
DEATH_RATE = 0.25/24.
SIM_RUNS = int(sys.argv[1]) # number of sims to run taken as command line arg
simulation = lib.simulation_ancestor_tracking
outdir = 'coop_neighbour_distribution/'
if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)



# run simulations in parallel 
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
df = pd.DataFrame(sum(pool.map(partial(run_sim),range(SIM_RUNS)),[]))
pool.close()
pool.join()
df.to_csv(outdir+'batch1',index=False)