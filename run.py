import numpy as np
import libs.pd_lib as lib
import libs.data as data
import libs.plot as vplt
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

N = 6
timend = 10
timestep = 1.

rand = np.random.RandomState()
b,c,DELTA = 10.,1.0,0.025
mutation_rate = 1e-3

simulation = lib.simulation_decoupled_update
game = lib.prisoners_dilemma_averaged
game_constants = (b,c)

history = lib.run_simulation(simulation,N,timestep,timend,rand,DELTA,game,(b,c))
                
# tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
# tissue.properties['type'] = np.zeros(N*N,dtype=int)
# tissue.age = np.zeros(N*N,dtype=float)
# history = lib.run(tissue, simulation(tissue,dt,10./dt,timestep/dt,rand,DELTA,game,game_constants,True),10./dt,timestep/dt)[-1]
#


# data.save_all(history,'test',12)
# vplt.save_mpg_torus(history, 'test0', index=None,key = "mutant", timestep=1.0)
    
# history = lib.run_simulation_size_dependent_with_neutral_mutants(N,timestep,timend,rand)
# import libs.plot as vplt
# vplt.save_mpg_torus(history,'test',key=None)
# data.save_all(history,'test',0)