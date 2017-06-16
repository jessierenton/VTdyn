import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import seaborn as sns
import copy
from initialisation import *

class Mesh(object):
    """ 
    Attributes: N_cells = number of cells 
                centres = array of (x,y) values for both cell and ghost node positions
                ghost_mask = boolean array such that ghost_mask = True if centre represents a cell, False if a ghost_node
                vnv = contains neighbour information for cells (extract using neighbours method)
                tri = (N_tot,3) array containing indices of centres forming triangles (used for plot_tri method)
    """
    
    def __init__(self,centres,ghost_mask):
        self.N_cells = sum(ghost_mask)
        self.N_tot = len(centres)
        self.centres = centres
        self.ghost_mask = ghost_mask
        self.vnv, self.tri = _delaunay(centres)
    
    def __len__(self):
        return self.N_tot
    
    def copy(self):
        return copy.deepcopy(self)
    
    def update(self):
        self.vnv, self.tri = _delaunay(self.centres)
        self.N_cells = sum(self.ghost_mask)
        self.N_tot = len(self.centres)
        
    def move(self, dr):
        self.centres = self.centres + dr
    
    def add(self,pos,gm=True):
        self.centres = np.append(self.centres,[pos],0)
        self.ghost_mask = np.append(self.ghost_mask,gm)
    
    def remove(self,idx):
        self.centres = np.delete(self.centres,idx,0)
        self.ghost_mask = np.delete(self.ghost_mask,idx)
    
    def neighbours(self,k):
        return self.vnv[1][self.vnv[0][k]:self.vnv[0][k+1]]
    
    def seperation(self,i,j):
        distance = np.sqrt(np.sum(((self.centres[i]-self.centres[j]))**2))      
        return distance, (self.centres[i]-self.centres[j])/distance
        
    def plot_tri(self,label=False):
        real_centres = self.centres[self.ghost_mask]
        ghost_centres = self.centres[~self.ghost_mask]        
        plt.triplot(self.centres[:,0], self.centres[:,1], self.tri.copy())
        plt.plot(real_centres[:,0], real_centres[:,1], 'o')
        plt.plot(ghost_centres[:,0],ghost_centres[:,1], 'o')
        if label:
            for i, coords in enumerate(self.centres):
                plt.text(coords[0],coords[1],str(i))
        plt.show()
        
    def voronoi(self,centres):
        return _voronoi(centres)
                    
       
def _voronoi(centres):
    return Voronoi(centres)
                                                  
def _delaunay(centres):                   
    tri = Delaunay(centres)
    return tri.vertex_neighbor_vertices, tri.simplices

if __name__ == '__main__':
    rand = np.random.RandomState(123456)
    centres, ghost_mask = init_centres(6,6,3,0.01,rand)
    mesh = Mesh(centres,ghost_mask)