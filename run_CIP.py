import numpy as np
import libs.contact_inhibition_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 1000. # simulation time (hours)
timestep = 96. # time intervals to save simulation history
init_time = None

rand = np.random.RandomState(128) 
simulation = lib.simulation_contact_inhibition_area_dependent  #simulation routine imported from lib
threshold_area_fraction=1.0
death_to_birth_rate_ratio = 0.1
DEATH_RATE = 0.25/24 #death rate: 0.25 deaths per day per cell
rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio) #death_rate,division_rate
domain_size_multiplier=0.948836

DELTA = 0.025
game_constants = (1.,1.)
game = lib.prisoners_dilemma_averaged

history = lib.run_simulation(simulation,l,timestep,timend,rand,progress_on=True,
            init_time=init_time,til_fix=False,save_areas=True,cycle_phase=None,
            return_events=False,save_cell_histories=False,domain_size_multiplier=domain_size_multiplier,
            DELTA=DELTA,game=game,game_constants=game_constants,mutant_num=100,
            rates=rates,threshold_area_fraction=threshold_area_fraction)
