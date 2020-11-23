import numpy as np
# from pathos.multiprocessing import cpu_count
# from pathos.pools import ParallelPool as Pool
import pandas as pd
from multiprocessing import Pool,cpu_count
import libs.public_goods_lib as lib #library for simulation routines
import libs.data as data
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import sys,os
from itertools import product
from functools import partial

DELTA = 0.025
L = 10 # population size N=l*l
TIMEND = 10000. # simulation time (hours)
TIMESTEP = 12. # time intervals to save simulation history
INIT_TIME = 12.


s,h,b = float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]) # command line args give params for logistic function
SIM_RUNS = int(sys.argv[4]) # number of sims to run taken as command line arg

PARENTDIR = 'neighbour_data_pgg/'
if not os.path.exists(PARENTDIR): # if the outdir doesn't exist create it
     os.makedirs(PARENTDIR)
savename ='h%.1f_s%03.0f_b%.0f'%(h,s,b)

game = lib.sigmoid_game
simulation = lib.simulation_decoupled_update

with open(PARENTDIR+'info',"w") as f:
    f.write('pop size = %3d\n'%(L*L))
    f.write('timestep = %.1f'%TIMESTEP)

def distribution_data(history,i):
    """
    generates neighbour data for cooperators and defectors (type 1 and 0)
        returns list of dicts with keys: tissueid, time, n, k, type
    n = # cooprators
    k = # neighbours
    """
    return [{'tissueid':i,'time':int(tissue.time),'n':sum(tissue.properties['type']),'k':len(cell_neighbours),'type':cell_type} 
            for tissue in history
                for cell_type,cell_neighbours in zip(tissue.properties['type'],tissue.mesh.neighbours)]    

def fixed(history):
    """returns True if cooperation fixates, otherwise returns False"""
    if 0 not in history[-1].properties['type']:
        return True 
    else:
        return False

def run_sim(i):
    """run a single simulation to fixation"""
    game_constants = (b,1.,s,h)
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,DELTA,game,game_constants,mutant_num=1,
                init_time=INIT_TIME,til_fix=True,save_areas=False,progress_on=False)
    if fixed(history):
        return distribution_data(history,i)

cpunum=cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
gen = pool.imap(partial(run_sim),range(10000))
i = 0 
results = []
for data in gen:
    if data is not None:
        results += data
        i += 1
        if i == SIM_RUNS:
            break
df = pd.DataFrame(results)
pool.close()
pool.join()
df.to_csv(PARENTDIR+savename,index=False)
