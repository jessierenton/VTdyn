import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import seaborn as sns
import copy



class Mesh(object):
    """ 
    Attributes: N_cells = number of cells 
                centres = array of (x,y) values for both cell and ghost node positions
                vnv = contains neighbour information for cells (extract using neighbours method)
                tri = (N_mesh,3) array containing indices of centres forming triangles (used for plot_tri method)
    """
    
    def __init__(self,centres):
        self.N_mesh = len(centres)
        self.centres = centres
        self.neighbours = self._get_neighbours()
    
    def __len__(self):
        return self.N_mesh
    
    def copy(self):
        return copy.deepcopy(self)
    
    def update(self):
        self.N_mesh = len(self.centres)
        self.neighbours = self._get_neighbours()
    
    def extreme_point(self):
        return np.max(self.distances())
    
    def distances(self):
        return np.array([np.sqrt(r[0]**2+r[1]**2) for r in self.centres])
        
    def move(self, i, dr):
        self.centres[i] += dr
    
    def add(self,pos):
        self.centres = np.append(self.centres,[pos],0)
    
    def remove(self,i):
        self.centres = np.delete(self.centres,i,0)
    
    def seperation(self,i,j_list):
        distance = np.sqrt(np.sum(((self.centres[i]-self.centres[j_list]))**2))      
        return distance, (self.centres[i]-self.centres[j_list])/distance
        
    def voronoi(self):
        return _voronoi(self.centres)
    
    def convex_hull(self):
        return _convex_hull(self.centres())
    
    def _get_neighbours(self):
        vnv = Delaunay(self.centres).vertex_neighbor_vertices
        neighbours = [(vnv[1][vnv[0][k]:vnv[0][k+1]]) for k in xrange(self.N_mesh)]
        return neighbours

class MeshTor(Mesh):
    
    def __init__(self,centres,width,height,N_mesh):
        self.N_mesh = N_mesh
        self.width,self.height = width,height
        self.centres = centres
        self.neighbours = self._get_neighbours()
    
    def copy(self):
        return copy.deepcopy(self)
    
    def update(self):
        self.neighbours = self._get_neighbours()
        self.N_mesh = len(self.centres)
        
    def move(self, i, dr):
        coords = self.periodise(self.centres[i] + dr)
        self.centres[i] = coords
    
    def periodise(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            if coords[i] >= L: coords[i] -= L
            elif coords[i] < L: coords[i] += L
        return coords
    
    def add(self,pos):
        self.periodise(pos)
        self.centres = np.append(self.centres,[pos],0)
    
    def seperation(self,i,j_list):
        """vector pointing from j to i"""
        p1,plist = self.centres[i],self.centres[j_list]
        dimensions = np.array(self.width, self.height)
        delta = np.abs(p1-plist)
        signs = np.sign(p1-plist)
        sep_vectors = np.where(delta > 0.5 * dimensions, (delta - dimensions)*signs, delta*signs)
        distance = np.linalg.norm(sep_vectors,axis=-1)       
        return distance, sep_vectors/distance
        
    def voronoi(self):
        return _voronoi(self.centres)
    
    def convex_hull(self):
        return _convex_hull(self.centres())  
    
    def _get_neighbours(self):
        centres,width,height = self.centres,self.width,self.height
        vnv = Delaunay(np.vstack([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]])).vertex_neighbor_vertices
        neighbours = [(vnv[1][vnv[0][k]:vnv[0][k+1]])%self.N_mesh for k in xrange(4*self.N_mesh,5*self.N_mesh)]
        return neighbours  
                    
def _convex_hull(centres):
    return ConvexHull(centres)
       
def _voronoi(centres):
    return Voronoi(centres)

