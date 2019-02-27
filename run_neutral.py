import numpy as np
import libs.pd_lib_neutral as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 50 # simulation time (hours)
timestep = 0.1 # time intervals to save simulation history

rand = np.random.RandomState()

simulation = lib.simulation_ancestor_tracking  #simulation routine imported from lib

history = lib.run_simulation(simulation,l,timestep,timend,rand,til_fix=False)
data.save_neighbour_distr(history,'EPS_1_test')