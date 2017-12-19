from multiprocessing import Process,Pool,Lock  #parallel processing
import multiprocessing as mp
from itertools import repeat


import os
import numpy as np
import libs.run_lib as lib
import libs.data as data
import libs.plot as vplt

rand = np.random.RandomState()

runs = 100
l = 12
timend = 10000.
timestep = 2.0

folder = 'multirun'

info = """
size-dependent selection with mutation for x0.9 spring constant
maxtime = %.0f
timestep = %.0f
N0 = %d
""" %(timend,timestep,l*l)

if not os.path.exists(folder): # if the folder doesn't exist create it
     os.makedirs(folder)

with open(folder+'/info.txt',"w") as infofile:
    infofile.write(info)

def run_parallel(i):
    rand=np.random.RandomState()
    history = lib.run_simulation_size_dependent_with_mutants(l,timestep,timend,rand)
    data.save_N_cell(history,folder,i)
    data.save_N_mutant(history,folder,i)
    if 0 not in history[-1].properties['mutant']: vplt.save_mpg_torus(history,folder, index=i,key = "mutant", timestep=2.)  
    
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum) # creating a pool with processors equal to the number of processors
pool.map(run_parallel,range(runs))  # mapping of all the calls necessary into the calling function