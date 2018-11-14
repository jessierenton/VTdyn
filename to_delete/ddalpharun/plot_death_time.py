import numpy as np
import libs.run_lib as lib
import libs.data as data
from structure.global_constants import *

rand = np.random.RandomState()

N = 10
timend = 1000.
timestep = 0.1
delta=0.1

history = lib.run_simulation_size_dependent(N,timestep,timend,rand)

data.save_all(history,'testfix',1)

histories = np.array(data.get_cell_histories())
fates = np.array([h['fate'] for h in histories],dtype=bool)
final_age_r = [cell['age'][-1] for cell in histories[fates]]
final_age_a = [cell['age'][-1] for cell in histories[~fates]]

