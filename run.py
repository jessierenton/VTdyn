import numpy as np
import libs.run_lib as lib
import libs.data as data

rand = np.random.RandomState(12345)

N = 12
timend = 200.
timestep = 1.


history = lib.run_simulation_size_dependent(N,timestep,timend,rand)
# history = lib.run_simulation_no_death(N,timestep,timend,rand)
# data.save_N_cell(history,'test',2)