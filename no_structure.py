import numpy as np
from structure.cell import Cell
from copy import deepcopy

dt = 0.005
rand = np.random.RandomState(42)

cell_ages = [0.0]
cycle_len = 10+rand.rand*4
for j in xrange(int(1e5)):
    time = j*dt
    cell_ages = np.array([cell.age for cell in cell_array[cell_live]])
    cell_aod = np.array([cell.aod for cell in cell_array[cell_live]])
    cell_cycle_len = np.array([cell.cycle_len for cell in cell_array[cell_live]])
    ready = np.where(cell_ages > cell_cycle_len)[0]
    dead = np.where(cell_ages > cell_aod)[0]
    
    for i in ready: 
        cell =cell_array[cell_live][i]
        N= len(cell_array)
        cell_array = np.append(cell_array,(Cell(rand,N,cell.id),Cell(rand,N+1,cell.id)))
        cell_live = np.append(cell_live, (True,True))
        cell_live[cell_live][i] = False
        N+=2
    cell_ages += dt
        
    cell_live[cell_live][dead] = False
    history.append(deepcopy(cell_array))