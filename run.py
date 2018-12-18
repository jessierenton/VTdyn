import numpy as np
import libs.pd_lib_density as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

"""run a single voronoi tessellation model simulation"""

l = 6 # population size N=l*l
timend = 100 # simulation time (hours)
timestep = 0.05 # time intervals to save simulation history

rand = np.random.RandomState()
b,c,DELTA = 3.,1.0,0.025 #prisoner's dilemma game parameters
OMEGA = 1.

simulation = lib.simulation_pd_density_dep  #simulation routine imported from lib
game = lib.prisoners_dilemma_averaged #game imported from lib
game_parameters = (b,c)


history = lib.run_simulation(simulation,l,timestep,timend,rand,DELTA,OMEGA,game,game_parameters,til_fix=False,mutant_num=1)
                