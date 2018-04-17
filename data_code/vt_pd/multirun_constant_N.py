from multiprocessing import Process,Pool,Lock  #parallel processing
import multiprocessing as mp
from itertools import repeat
import sys
from functools import partial

import os
import numpy as np
import libs.run_lib as lib
import libs.data as data
# import libs.plot as vplt
from libs.run_lib import run_simulation_poisson_const_pop_size,prisoners_dilemma_averaged,prisoners_dilemma_accumulated
from functools import partial

game_str = sys.argv[1]
if game_str == 'acc': game = prisoners_dilemma_accumulated
elif game_str == 'av': game = prisoners_dilemma_averaged
else: raise ValueError('invalid game string')

b_vals = np.array(sys.argv[4:],dtype=float)
#prisoner dilemma params
c,DELTA = 1.0,0.025

rand = np.random.RandomState()
outdir = 'delta%.3f_pd_%s'%(DELTA,game_str)

l = 10
timend = 10000.
timestep = 12.0

start_batch,end_batch = int(sys.argv[2]),int(sys.argv[3])
runs_per_batch = 1000


width, height = float(l)*1.5, float(l)*1.5*np.sqrt(3)/2

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

def run_parallel(b,i):
    rand=np.random.RandomState()
    history = lib.run_simulation_poisson_const_pop_size(l,timestep,timend,rand,DELTA,prisoners_dilemma_averaged,(b,c),save_areas=False)
    if 0 not in history[-1].properties['type']:
        fix = 1  
        # data.save_N_mutant_type(history,outdir+'/fixed_b%.1f'%b,i)
    elif 1 not in history[-1].properties['type']:
        fix = 0
    else: 
        fix = -1
        data.save_N_mutant_type(history,outdir+'/incomplete_b%.1f'%b,i)
    return fix

cpunum=mp.cpu_count()
pool = Pool(processes=cpunum,maxtasksperchild=1000) # creating a pool with processors equal to the number of processors
for b in b_vals:
    fix_results = open(outdir+'/fix_b%.1f'%b,'w',0)
    for i in range(start_batch,end_batch):
        text = '\r running batch %d of %d'%(i+1,end_batch)
        sys.stdout.write(text)
        sys.stdout.flush()
        fixation = np.array([f for f in pool.imap(partial(run_parallel,b),range(i*runs_per_batch,(i+1)*runs_per_batch))])  # mapping of all the calls necessary into the calling function
        fixed = len(np.where(fixation==1)[0])
        nofixed = len(np.where(fixation==0)[0])
        incomplete = len(np.where(fixation==-1)[0])
        fix_results.write('%d    %d    %d\n'%(fixed,nofixed,incomplete))
