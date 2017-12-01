from multiprocessing import Process,Pool,Lock  #parallel processing
import multiprocessing as mp
from itertools import repeat


import os
import numpy as np
import libs.run_lib as lib
import libs.data as data

runs = 3
rand = np.random.RandomState()

l = 20
timend = 50.
timestep = 0.5

folder = 'run2'

info = """
death: delayed uniform (mean 12)
birth: delayed uniform (mean 12, delay 10)
timend = %.2f
timestep = %.2f
N = %d

""" %(timend,timestep,l*l)

if not os.path.exists(folder): # if the folder doesn't exist create it
     os.makedirs(folder)

with open(folder+'/info.txt',"w") as infofile:
    infofile.write(info)

def run_parallel(i):
    rand=np.random.RandomState()
    history = lib.run_simulation_poisson_death_and_div(l,timestep,timend,rand)
    data.save_N_cell(history,folder,i)
    
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum) # creating a pool with processors equal to the number of processors
pool.map(run_parallel,range(runs))  # mapping of all the calls necessary into the calling function