import numpy as np
import libs.run_lib as lib
import libs.data as data
from structure.global_constants import *

rand = np.random.RandomState(42)

N = 12
timend = 50.
timestep = 1.


history = lib.run_simulation_size_dependent(N,timestep,timend,rand)
# history = lib.run_simulation_no_death(N,timestep,timend,rand)
# data.save_N_cell(history,'test',2)

# data.save_division_times(history,'times',int(-MU))