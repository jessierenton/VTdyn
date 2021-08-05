import numpy as np
import libs.pd_lib_neutral as lib
import libs.data as data
import libs.plot as vplt #plotting library
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import pandas as pd

"""run a single voronoi tessellation model simulation"""

def distribution_data(history,mutant_id,i,all_types=False):
    """
    generates neighbour data for mutants (or all cells if all_types is True)
    cells are labelled by their ancestor. all cells with ancestor=mutant_id are type 1, all other cells type 0.
        returns list of dicts with keys: tissueid, time, n, k, j [, type] 
    n = # type 1 cells
    k = # neighbours
    j = # type 1 neighbours
    """
    if all_types:
        return [{'tissueid':i,'time':int(tissue.time),'n':sum(tissue.properties['ancestor']==mutant_id),'k':len(cell_neighbours),
                'j':sum((tissue.properties['ancestor']==mutant_id)[cell_neighbours]),'type': 1 if tissue.properties['ancestor'][idx]==mutant_id else 0} 
                for tissue in history if 1<=sum(tissue.properties['ancestor']==mutant_id)<100
                    for idx,cell_neighbours in enumerate(tissue.mesh.neighbours)]    
    else:
        return [{'tissueid':i,'time':int(tissue.time),'n':sum(tissue.properties['ancestor']==mutant_id),'k':len(cell_neighbours),
                'j':sum((tissue.properties['ancestor']==mutant_id)[cell_neighbours])} 
                for tissue in history if 1<=sum(tissue.properties['ancestor']==mutant_id)<100
                    for idx,cell_neighbours in enumerate(tissue.mesh.neighbours) if tissue.properties['ancestor'][idx]==mutant_id]     




L = 10 # population size N=l*l
timend = 10000 # simulation time (hours)
timestep = 12.0 # time intervals to save simulation history
init_time = 12.
rand = np.random.RandomState()

simulation = lib.simulation_ancestor_tracking # tracks clones with common ancestor



rand = np.random.RandomState()
history = lib.run_simulation(simulation,L,timestep,timend,rand,progress_on=True,
                            init_time=init_time,til_fix=True,save_areas=False)
