import numpy as np
import libs.contact_inhibition_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

PROGRESS_ON = True
SAVE_CELL_HISTORIES = False
TIL_FIX = False
DEATH_RATE = 0.25/24 #death rate: 0.25 deaths per day per cell

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 24 # simulation time (hours)
timestep = 6 # time intervals to save simulation history
init_time = 96

rand = np.random.RandomState() 
# simulation = lib.simulation_contact_inhibition_area_dependent  #simulation routine imported from lib
simulation = lib.simulation_contact_inhibition_area_dependent
threshold_area_fraction,death_to_birth_rate_ratio,domain_size_multiplier = 1.200000,0.100000,1.030535
# threshold_area_fraction,death_to_birth_rate_ratio,domain_size_multiplier=0.800000,0.100000,0.859628
rates = (DEATH_RATE,DEATH_RATE/death_to_birth_rate_ratio*1.125) #death_rate,division_rate


DELTA = 0.025
game_constants = (4.,1.)
game = lib.prisoners_dilemma_averaged
initial_mutants = 1

history = lib.run_simulation(simulation,l,timestep,timend,rand,progress_on=PROGRESS_ON,
            init_time=init_time,til_fix=TIL_FIX,save_areas=True,cycle_phase=None,
            return_events=False,save_cell_histories=SAVE_CELL_HISTORIES,domain_size_multiplier=domain_size_multiplier,
            DELTA=DELTA,game=game,game_constants=game_constants,mutant_num=initial_mutants,mutant_type=1,ancestors=True,
            rates=rates,threshold_area_fraction=threshold_area_fraction)


# parameters = {'width':10*domain_size_multiplier,'db':death_to_birth_rate_ratio,'alpha':threshold_area_fraction}
# data.save_as_json(history,'fate_test_cip',['cell_histories'],parameters,index=0)
  