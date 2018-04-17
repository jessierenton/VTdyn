import numpy as np
import libs.plot as vplt
from structure.global_constants import *
import libs.run_lib as lib

import numpy as np
import libs.run_lib as lib
import libs.data as data
import libs.plot as vplt
from structure.global_constants import *


N = 10
timend = 100
timestep = 1.0

rand = np.random.RandomState()
b,c,DELTA = 10.,1.0,0.0

def get_graph(tissue):
    neighbours=tissue.mesh.neighbours
    g = igraph.Graph()
    g.add_vertices(len(neighbours))
    for i,j in enumerate(neighbours):
        g.add_edge(i,j)

def get_adj_matrix(neighbours):
    n = len(neighbours)
    adj_mat = np.zeros((n,n),dtype=float)
    for cell,cell_neighbours in enumerate(neighbours):
        adj_mat[cell][cell_neighbours] = 1
    return adj_mat


from solve_ctimes import find_critical_ratio
histories = [lib.run_simulation_poisson_const_pop_size(N,timestep,timend,rand,(b,c,DELTA),save_areas=False) for i in range(10)]
crit_ratios = [find_critical_ratio(get_adj_matrix(history[i].mesh.neighbours)) for i in np.arange(5)*20 for history in histories]
print 'mean = %.4f, stdev = %.4f' %(np.mean(crit_ratios,np.std(crit_ratios))) 