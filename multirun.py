from multiprocessing import Process,Pool,Lock  #parallel processing
import multiprocessing as mp
from itertools import repeat
import sys

import os
import numpy as np
import libs.run_lib as lib
import libs.data as data
# import libs.plot as vplt
from libs.run_lib import run_simulation_size_dependent #,simulation_size_dependent

rand = np.random.RandomState()
outdir = 'test'

l = 10
timend = 5.
timestep = 1.0

width, height = float(l)*1.5, float(l)*1.5*np.sqrt(3)/2

if not os.path.exists(outdir): # if the folder doesn't exist create it
     os.makedirs(outdir)


def run_parallel(i):
    rand=np.random.RandomState()
    history = run_simulation_size_dependent(l,timestep,timend,rand)
    data.save_all(history,'%s/T_d%02d'%(outdir,int(T_d)),i)

cpunum=mp.cpu_count()
pool = Pool(processes=cpunum,maxtasksperchild=1000) # creating a pool with processors equal to the number of processors
for newT_d in (np.arange(4,dtype=float)*2+12):
    global T_d; T_d = newT_d
    [f for f in pool.imap(run_parallel,range(2))]  # mapping of all the calls necessary into the calling function

