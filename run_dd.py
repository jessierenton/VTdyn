import numpy as np
import libs.density_dep_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 10 # simulation time (hours)
timestep = 1. # time intervals to save simulation history

rand = np.random.RandomState(42)
game = "simple"
game_params = 1.
DELTA=0.0
birth_dd_func = lib.step_density_dep
death_dd_func = lib.step_density_dep
birth_dd_params = (7*2./3**0.5,True) 
death_dd_params = (7*2./3**0.5,False) 
birth_to_death_rate_ratio = 1.

params={'game':game,'game_params':game_params,'DELTA':DELTA,'birth_dd_func':birth_dd_func,'death_dd_func':death_dd_func,
            'birth_dd_params':birth_dd_params,'death_dd_params':death_dd_params, 'birth_to_death_rate_ratio':birth_to_death_rate_ratio}

simulation = lib.simulation_local_density_dep

history = lib.run_simulation(simulation,l,timestep,timend,rand,progress_on=True,til_fix=False,
            init_time=None,save_areas=True,store_dead=True,save_events=True,**params)
                