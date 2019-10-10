import numpy as np
import libs.contact_inhibition_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 100. # simulation time (hours)
timestep = 1. # time intervals to save simulation history

rand = np.random.RandomState()

simulation = lib.simulation_contact_inhibition_area_dependent  #simulation routine imported from lib
threshold_area_fraction=0.99
rates = (1./24,1./12) #death_rate,division_rate
# domain_size_multiplier=np.sqrt(0.8)
domain_size_multiplier=1

history = lib.run_simulation(simulation,l,timestep,timend,rand,progress_on=True,
            init_time=None,til_fix=False,save_areas=True,cycle_phase=None,
            return_events=False,save_cell_histories=True,domain_size_multiplier=domain_size_multiplier,
            rates=rates,threshold_area_fraction=threshold_area_fraction)

