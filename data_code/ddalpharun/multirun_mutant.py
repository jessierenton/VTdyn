from multiprocessing import Process,Pool,Lock  #parallel processing
import multiprocessing as mp
from itertools import repeat
from functools import partial
import sys

import os
import numpy as np
import libs.run_lib as lib
import libs.data as data
import libs.plot as vplt

rand = np.random.RandomState()

runs_per_batch = 4
batches = 2
l = 4
timend = 10.
timestep = 1.0

width, height = float(l)*1.5, float(l)*1.5*np.sqrt(3)/2

folder = 'mutants'

info = """
size-dependent selection with neutral mutation
maxtime = %.0f
timestep = %.0f
N0 = %d
box height = %.2f
box width = %.2f

""" %(timend,timestep,l*l,width,height)

if not os.path.exists(folder): # if the folder doesn't exist create it
     os.makedirs(folder)

with open(folder+'/info.txt',"w",0) as infofile:
    infofile.write(info)

def run_parallel(update_file,i):
    alpha = 0.25
    rand=np.random.RandomState()
    history = lib.run_simulation_size_dependent_with_mutants(alpha,l,timestep,timend,rand)
    if 0 not in history[-1].properties['mutant']:
        fix = 1 
        # data.save_N_cell(history,folder+'/fixed',i)
        # data.save_N_mutant(history,folder+'/fixed',i)
        import libs.plot
        save_mpg_torus(history, name, index=None,key = "mutants", timestep=0.5)
    elif 1 not in history[-1].properties['mutant']:
        fix = 0
    else: 
        fix = -1
        # data.save_N_cell(history,folder+'/incomplete',i)
        # data.save_N_mutant(history,folder+'/incomplete',i)
    update_file.write('%d\n'%i)
    return fix

fix_results = open(folder+'/fixation.txt','w',0)
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum,maxtasksperchild=1000) # creating a pool with processors equal to the number of processors
for i in range(batches):
    text = '\rbatch %d of %d'%(i+1,batches)
    sys.stdout.write(text)
    sys.stdout.flush()
    update_file = open(folder+'/current_batch','w',0)
    fixation = np.array(pool.imap(partial(run_parallel,update_file),range(i*runs_per_batch,(i+1)*runs_per_batch)))  # mapping of all the calls necessary into the calling function
    fixed = len(np.where(fixation==1)[0])
    nofixed = len(np.where(fixation==0)[0])
    incomplete = len(np.where(fixation==-1)[0])
    fix_results.write('%d    %d    %d\n'%(fixed,nofixed,incomplete) )
    update_file.close()
pool.join()
pool.close()