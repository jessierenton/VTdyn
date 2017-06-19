import numpy as np
from scipy.spatial import Delaunay, Voronoi, ConvexHull
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import seaborn as sns
import copy

class Mesh(object):
    """ 
    Attributes: N_cells = number of cells 
                centres = array of (x,y) values for cell centre positions in mesh
                vnv = contains neighbour information for cells (extract using neighbours method)
                tri = (N_tot,3) array containing indices of centres forming triangles (used for plot_tri method)
    """
    
    def __init__(self,centres):
        self.N_tot = len(centres)
        self.centres = centres
        self.vnv, self.tri = _delaunay(centres)
    
    def __len__(self):
        return self.N_tot
    
    def copy(self):
        return copy.deepcopy(self)
    
    def update(self):
        self.vnv, self.tri = _delaunay(self.centres)
        self.N_tot = len(self.centres)
        
    def move(self, dr):
        self.centres = self.centres + dr
    
    def add(self,pos):
        self.centres = np.append(self.centres,[pos],0)
    
    def remove(self,idx):
        self.centres = np.delete(self.centres,idx,0)
    
    def neighbours(self,k):
        return self.vnv[1][self.vnv[0][k]:self.vnv[0][k+1]]
    
    def seperation(self,i,j):
        distance = np.sqrt(np.sum(((self.centres[i]-self.centres[j]))**2))      
        return distance, (self.centres[i]-self.centres[j])/distance
        
    def plot_tri(self,label=False):    
        plt.triplot(self.centres[:,0], self.centres[:,1], self.tri.copy())
        plt.plot(self.centres[:,0], self.centres[:,1], 'o')
        if label:
            for i, coords in enumerate(self.centres):
                plt.text(coords[0],coords[1],str(i))
        plt.show()
        
    def voronoi(self):
        return _voronoi(self.centres)
    
    def convex_hull(self):
        return _convex_hull(self.centres())
                    
def _convex_hull(centres):
    return convex_hull(centres)
       
def _voronoi(centres):
    return Voronoi(centres)
                                                  
def _delaunay(centres):                   
    tri = Delaunay(centres)
    return tri.vertex_neighbor_vertices, tri.simplices

if __name__ == '__main__':
    rand = np.random.RandomState(123456)
    centres = init_centres(6,6,3,0.01,rand)
    mesh = Mesh(centres)