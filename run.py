import numpy as np
import libs.run_lib as lib
import libs.data as data

rand = np.random.RandomState()

N = 10
timend = 20.
timestep = 1.0


history = lib.run_simulation_death_and_div(N,timestep,timend,rand)
# history = lib.run_simulation_no_death(N,timestep,timend,rand)
# data.save_N_cell(history,'test',2)