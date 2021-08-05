import numpy as np
import libs.pd_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 96. # simulation time (hours)
timestep = 2. # time intervals to save simulation history

rand = np.random.RandomState()
b,c,DELTA = 6.,1.0,1 #prisoner's dilemma game parameters

simulation = lib.simulation_decoupled_update_exp_fitness  #simulation routine imported from lib
game = lib.prisoners_dilemma_averaged #game imported from lib
game_parameters = (b,c)


history = lib.run_simulation(simulation,l,timestep,timend,rand,DELTA,game,game_parameters,til_fix=False,init_time=12,save_areas=False,progress_on=True,mutant_num=10)
                