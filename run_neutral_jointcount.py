import numpy as np
import libs.pd_lib_neutral as lib
import libs.data as data
import libs.plot as vplt #plotting library
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import pandas as pd
from multiprocessing import Pool

"""run a single voronoi tessellation model simulation"""  

def adjacency_matrix(tissue):
    A = np.zeros((len(tissue),len(tissue)),dtype=int)
    for cell,cell_neighbours in enumerate(tissue.mesh.neighbours):
        A[cell][cell_neighbours] = 1
    return A

def add_cell_types(history,cell_id=None):
    if cell_id is None:
        cell_id = np.argmax(np.bincount(history[-1].properties['ancestor']))
    for tissue in history:
        tissue.properties['type'] = (tissue.properties['ancestor']==cell_id)*1

def joint_count_stats(adjmatrix,types):
    N = len(types)
    MM = 0.5*sum(types[i]*types[j]*adjmatrix[i,j] 
            for i in range(N) for j in range(N))
    WW = 0.5*sum((1-types[i])*(1-types[j])*adjmatrix[i,j] 
            for i in range(N) for j in range(N))
    MW = 0.5*sum((types[i]-types[j])**2*adjmatrix[i,j]
            for i in range(N) for j in range(N))
    return MM,WW,MW

def get_join_count_stats_history(history,randomise_types=False):
    if randomise_types:
        stats = np.array([joint_count_stats(adjacency_matrix(tissue),
                        rand.permutation(tissue.properties['type']))
                    for tissue in history])
    else:
        stats = np.array([joint_count_stats(adjacency_matrix(tissue),
                        tissue.properties['type'])
                    for tissue in history])

    n_mutant = np.array([sum(tissue.properties['type']) for tissue in history])
    return stats,n_mutant

def joint_count_multi_df(histories):
    jc_dfs = [joint_count_df(history).assign(sim=i) 
                for i,history in enumerate(histories)]
    return pd.concat(jc_dfs,ignore_index=True)

def joint_count_df(history):
    add_cell_types(history)
    jc,n = get_join_count_stats_history(history,False)
    random_jc,n1 = get_join_count_stats_history(history,True)
    print(np.all(n==n1))
    return create_df(jc,random_jc,n)

def create_df(jc,random_jc,n):
    nvals = np.tile(n,2)
    types = ['clustered']*len(n)+['random']*len(n)
    CC,DD,CD = [np.hstack([jc.T[i],random_jc.T[i]]) for i in range(3)]
    df = pd.DataFrame({'n':nvals,'types':types,'CC':CC,'DD':DD,'CD':CD})
    return df

def randomise_types(history):
    i = history[-1].properties['ancestor'][0]
    
def run_sim(i):
    rand = np.random.RandomState()
    return lib.run_simulation(simulation,L,timestep,timend,
                            rand,progress_on=True,init_time=init_time,
                            til_fix=True,save_areas=False)

if __name__ == "__main__":
    L = 10 # population size N=l*l
    timend = 10000 # simulation time (hours)
    timestep = 1. # time intervals to save simulation history
    init_time = 12.
    rand = np.random.RandomState()

    simulation = lib.simulation_ancestor_tracking # tracks clones with common ancestor
    pool = Pool(maxtasksperchild=1000)

    histories = [history for history in pool.imap(run_sim,range(10))]
    df = joint_count_multi_df(histories)
    df.to_csv('jointcount',ignore_index=True)
        
