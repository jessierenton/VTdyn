import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull
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
    
    def extreme_point(self):
        return np.max(self.distances())
    
    def distances(self):
        return np.array([np.sqrt(r[0]**2+r[1]**2) for r in self.centres])
        
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
        
    def voronoi(self):
        return _voronoi(self.centres)
    
    def convex_hull(self):
        return _convex_hull(self.centres())
                    
def _convex_hull(centres):
    return ConvexHull(centres)
       
def _voronoi(centres):
    return Voronoi(centres)
                                                  
def _delaunay(centres):                   
    tri = Delaunay(centres)
    return tri.vertex_neighbor_vertices, tri.simplices

if __name__ == '__main__':
    rand = np.random.RandomState(123456)
    centres, ghost_mask = init_centres(6,6,3,0.01,rand)
    mesh = Mesh(centres,ghost_mask)