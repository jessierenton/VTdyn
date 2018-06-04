import numpy as np
import libs.pd_lib as lib
import libs.data as data
import libs.plot as vplt
from structure.global_constants import *

N = 10
timend = 1000
timestep = 1.

rand = np.random.RandomState()
b,c,DELTA = 10.,1.0,0.025
mutation_rate = 1e-3

simulation = lib.simulation_with_mutation
game = lib.prisoners_dilemma_accumulated
constants = (mutation_rate,b,c)

def initiate_cluster(n):
    mutants = [rand.rand(N*N)]
    for i in range(n-1):
        while True:
            cell = rand.choice(mutants)
            allowed = [neighbour for neighbour in tissue.mesh.neighbours[cell] if neighbour not in mutants]
            if len(allowed)>0: mutants += rand.choice(allowed); break
        
                

history = lib.run_simulation(simulation,N,timestep,timend,rand,DELTA,game,constants,False)



# data.save_all(history,'test',12)
# vplt.save_mpg_torus(history, 'test0', index=None,key = "mutant", timestep=1.0)
    
# history = lib.run_simulation_size_dependent_with_neutral_mutants(N,timestep,timend,rand)
# import libs.plot as vplt
# vplt.save_mpg_torus(history,'test',key=None)
# data.save_all(history,'test',0)