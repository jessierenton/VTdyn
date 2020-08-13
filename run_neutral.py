import numpy as np
import libs.pd_lib_neutral as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import pandas as pd

"""run a single voronoi tessellation model simulation"""

def get_coop_neighbour_distribution(history,coop_id):
    k_min = 3
    k_max = max(len(neighbours) for tissue in history
                                    for neighbours in tissue.mesh.neighbours)
    coop_neighbour_distribution = [[[0 for n in range(1,len(history[0]))] for j in range(0,k+1)] for k in range(k_min,k_max+1)]
    for tissue in history:
        cooperators = tissue.properties['ancestor']==coop_id
        n = sum(cooperators)
        for cell_neighbours in tissue.mesh.neighbours:
            k = len(cell_neighbours)
            j = sum(cooperators[cell_neighbours])
            coop_neighbour_distribution[k-k_min][j][n] += 1
    return coop_neighbour_distribution

def get_number_coop_neighbours(cell_neighbours,cooperators):
    return len(cell_neighbours),sum(cooperators[cell_neighbours])

def get_frequency(distribution):
    return np.nan_to_num(distribution/distribution.sum(axis=1,keepdims=True),0)

def get_distribution(tissue_data,k_min,k_max):
    tissue_data = np.array(tissue_data).T
    coop_coop_neighbour_distribution = [np.bincount((tissue_data[1])[tissue_data[0]==k]) 
                                            for k in range(k_min,k_max+1)]
    coop_coop_neighbour_distribution = np.array([np.pad(dist,(0,k_max+1-len(dist)),'constant') 
                                            for dist in coop_coop_neighbour_distribution],dtype=float)
    return get_frequency(coop_coop_neighbour_distribution) 

def get_coop_neighbour_distribution(history,coop_id):
    k_min = 3
    k_max = max(len(neighbours) for tissue in history
                                    for neighbours in tissue.mesh.neighbours)
    data = [(sum(tissue.properties['ancestor']==coop_id),[get_number_coop_neighbours(tissue.mesh.neighbours[cell_id],tissue.properties['ancestor']==coop_id)
        for cell_id in np.where(tissue.properties['ancestor']==coop_id)[0]]) for tissue in history]
    return [(n,get_distribution(tissue_data,k_min,k_max)) for n,tissue_data in data]
         
def average_distributions(coop_coop_dist_tissues):
    n_vals = sorted(set([dist[0] for dist in coop_coop_dist_tissues]))
    averaged_coop_coop_distr = [[np.mean(np.stack([dist[1] for dist in coop_coop_dist_tissues 
                                                                if dist[0]==n]),axis=0)] for n in n_vals]
    return np.vstack(averaged_coop_coop_distr)

def get_probs(distribution,n=None,k=None,j=None,k_min=3,n_min=1):
    if n is None:
        return distribution[:,k-k_min,j]
    elif k is None:
        return distribution[n-n_min,:,j]
    elif j is None:
        return distribution[n-n_min,k-k_min]
    else:
        return distribution[n-n_min,k-k_min,j]
        
def distribution_df(history,coop_id):
    data = [{'time':int(tissue.time),'n':sum(tissue.properties['ancestor']==coop_id),'k':len(cell_neighbours),
                'j':sum((tissue.properties['ancestor']==coop_id)[cell_neighbours])} 
                for tissue in history if 1<sum(tissue.properties['ancestor']==coop_id)<100
                    for idx,cell_neighbours in enumerate(tissue.mesh.neighbours) if tissue.properties['ancestor'][idx]==coop_id]
    return pd.DataFrame(data)

l = 10 # population size N=l*l
timend = 30 # simulation time (hours)
timestep = 1.0 # time intervals to save simulation history
init_time = 20
rand = np.random.RandomState()

simulation = lib.simulation_ancestor_tracking  #simulation routine imported from lib

history = lib.run_simulation(simulation,l,timestep,timend,rand,init_time,til_fix=True,save_areas=False,progress_on=True)
coop_id = np.argmax(np.bincount(history[-1].properties['ancestor']))
df = distribution_df(history,coop_id)