import numpy as np
import libs.run_lib as lib
import libs.data as data
import libs.plot as vplt
from structure.global_constants import *

N = 4
timend = 40
timestep = 1.

rand = np.random.RandomState()
b,c,DELTA = 10.,1.0,0.025


history = lib.run_simulation_poisson_const_pop_size(N,timestep,timend,rand,(b,c,DELTA),save_areas=False)


# data.save_all(history,'test',12)
# vplt.save_mpg_torus(history, 'test0', index=None,key = "mutant", timestep=1.0)
    
# history = lib.run_simulation_size_dependent_with_neutral_mutants(N,timestep,timend,rand)
# import libs.plot as vplt
# vplt.save_mpg_torus(history,'test',key=None)
# data.save_all(history,'test',0)