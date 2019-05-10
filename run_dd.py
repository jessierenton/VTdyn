import numpy as np
import libs.density_dep_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import sys
from optparse import OptionParser

parser = OptionParser()
parser.set_defaults(save_events=False,progress_on=False,til_fix=False,seed=None)
parser.add_option("-e","--events", action="store_true", dest="save_events",
                    help="store tissue every time a birth/death event occurs")
parser.add_option("-p","--progress", action="store_true", dest="progress_on",
                    help="show progress bar")
parser.add_option("-f","--fixation", action="store_true",dest="til_fix", 
                    help="stop simulation after mutants have fixed or died out")
parser.add_option("-s","--seed", type="int",dest="seed", metavar="SEED",
                    help="supply seed for random state")
                    
(options,args) = parser.parse_args()

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = float(args[0]) # simulation time (hours)
timestep = 1. # time intervals to save simulation history

rand = np.random.RandomState(options.seed)
game = "simple"
game_params = 1.
DELTA=0.0

# birth_dd_func = lib.step_density_dep
# death_dd_func = lib.step_density_dep
# birth_dd_params = (3.,7*2./3**0.5,True)
# death_dd_params = (3.,7*2./3**0.5,False)
birth_dd_func = lib.linear_density_dep
birth_dd_params = (-1.,7*2./3**0.5,1.2) 
death_dd_func = lib.no_density_dep
death_dd_params = (1,)

birth_to_death_rate_ratio = 1.

params={'game':game,'game_params':game_params,'DELTA':DELTA,'birth_dd_func':birth_dd_func,'death_dd_func':death_dd_func,
            'birth_dd_params':birth_dd_params,'death_dd_params':death_dd_params, 'birth_to_death_rate_ratio':birth_to_death_rate_ratio}

simulation = lib.simulation_local_density_dep

history = lib.run_simulation(simulation,l,timestep,timend,rand,progress_on=options.progress_on,til_fix=options.til_fix,
            init_time=None,save_areas=True,store_dead=True,save_events=options.save_events,**params)
                