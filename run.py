import numpy as np
import libs.run_lib as lib
import libs.data as data
import libs.plot as vplt
from structure.global_constants import *

N = 6
timend = 10
timestep = 1.0

rand = np.random.RandomState()


history = lib.run_simulation_size_dependent(N,timestep,timend,rand)

# vplt.save_mpg_torus(history, 'test0', index=None,key = "mutant", timestep=1.0)
    
# history = lib.run_simulation_size_dependent_with_neutral_mutants(N,timestep,timend,rand)
data.save_all(history,'test',0)
# import libs.plot as vplt
# vplt.save_mpg_torus(history,'long',key='ancestor')