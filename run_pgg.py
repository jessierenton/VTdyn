import numpy as np
import libs.public_goods_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import pandas as pd

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 200 # simulation time (hours)
timestep = 1.0 # time intervals to save simulation history
init_time = 96.
rand = np.random.RandomState()

simulation = lib.simulation_decoupled_update  #simulation routine imported from lib
DELTA = 0.025
b,c = 5.,1.
game = lib.N_person_prisoners_dilemma
game_constants = (b,c)


history = lib.run_simulation(simulation,l,timestep,timend,rand,DELTA,game,game_constants,mutant_num=5,
                init_time=init_time,til_fix=True,save_areas=False,progress_on=True)
