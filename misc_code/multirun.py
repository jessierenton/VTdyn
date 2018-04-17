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
from functools import partial

rand = np.random.RandomState()
outdir = 'test'

l = 10
timend = 500.
timestep = 1.0

width, height = float(l)*1.5, float(l)*1.5*np.sqrt(3)/2

if not os.path.exists(outdir): # if the folder doesn't exist create it
     os.makedirs(outdir)


def run_parallel(i):
    rand=np.random.RandomState()
    history = lib.run_simulation_poisson(N,timestep,timend,rand,save_areas=False)
    data.save_all(history,'%s/'%(outdir),i)

cpunum=mp.cpu_count()
pool = Pool(processes=cpunum,maxtasksperchild=1000) # creating a pool with processors equal to the number of processors
[f for f in pool.imap(partial(run_parallel),range(20))]  # mapping of all the calls necessary into the calling function

