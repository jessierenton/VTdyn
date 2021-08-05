from multiprocessing import Pool #parallel processing
from itertools import repeat
import sys
from functools import partial

import os
import numpy as np
import libs.data as data
from libs.pd_lib import run_simulation,simulation_decoupled_update,prisoners_dilemma_averaged,prisoners_dilemma_accumulated
from functools import partial

"""command line arguments
1. str (av or acc): game_str. determines payoff accounting is averaged or accumulated
2, 3. ints: start_batch and end_batch. indices (run batches of 1000 simulations)
4, ... array floats: b_vals. values of b (prisoner's dilemma param) to run simulations for 
"""
game_str = sys.argv[1]
if game_str == 'acc': game = prisoners_dilemma_accumulated
elif game_str == 'av': game = prisoners_dilemma_averaged
else: raise ValueError('invalid game string')

runs = 2

b = 3.0
c,DELTA = 1.0,0.025 #prisoner dilemma params

l = 20 #population size N=lxl 
timend = 10000. #time (hours) after which simulation ends if no fixation
timestep = 12.0 #state saved every 12 hours

rand = np.random.RandomState()

outdir = 'VTpd_%s_decoupled_long'%(game_str)
if not os.path.exists(outdir): # if the outdir doesn't exist create it
     os.makedirs(outdir)
with open(outdir+'/info','w') as f:
    f.write('N=%d, c=%.1f, delta=%.3f'%(l*l,c,DELTA))

def run_parallel(b,i):
    """run a single simulation using simulation_decoupled_update routine 
    """
    rand=np.random.RandomState()
    history = run_simulation(simulation_decoupled_update,l,timestep,timend,rand,DELTA,prisoners_dilemma_averaged,(b,c),save_areas=False,til_fix=False,mutant_num=60)
    data.save_N_mutant(history,outdir+'/l20_b%.1f'%b,i)
    return history

pool = Pool(maxtasksperchild=1000) # creating a pool of workers to run simulations in parallel

histories = [f for f in pool.imap(partial(run_parallel,b),range(runs))]  # mapping of all the calls necessary 
