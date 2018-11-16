import numpy as np
import libs.pd_lib as lib
import libs.data as data
import libs.plot as vplt
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

N = 4
timend = 10
timestep = 1.

rand = np.random.RandomState(42)
b,c,DELTA = 10.,1.0,0.025

simulation = lib.simulation_decoupled_update
game = lib.prisoners_dilemma_averaged
game_constants = (b,c)

history = lib.run_simulation(simulation,N,timestep,timend,rand,DELTA,game,game_constants)
                