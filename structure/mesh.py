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
        self.neighbours,self.distances,self.unit_vecs = self.get_neighbours()
    
    def __len__(self):
        return self.N_mesh
    
    def copy(self):
        meshcopy = copy.copy(self)
        meshcopy.centres = copy.copy(meshcopy.centres)
        return meshcopy
    
    def update(self):
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs = self.get_neighbours()
    
    def extreme_point(self):
        return np.max(self.distances())
    
    def distances(self):
        return np.array([np.sqrt(r[0]**2+r[1]**2) for r in self.centres])
        
    def move(self, i, dr):
        self.centres[i] += dr
    
    def move_all(self, dr_array):
        self.centres = self.centres + dr_array
    
    def add(self,pos):
        self.centres = np.append(self.centres,[pos],0)
    
    def remove(self,i):
        self.centres = np.delete(self.centres,i,0)
    
    def seperation(self,i,j_list):
        distance = np.sqrt(np.sum(((self.centres[i]-self.centres[j_list]))**2))      
        return distance, (self.centres[i]-self.centres[j_list])/distance
        
    def voronoi(self):
        return Voronoi(self.centres)
    
    def convex_hull(self):
        return ConvexHull(self.centres())
    
    def get_neighbours(self):
        centres = self.centres
        N_mesh = self.N_mesh
        vnv = Delaunay(centres).vertex_neighbor_vertices
        neighbours = [vnv[1][vnv[0][k]:vnv[0][k+1]] for k in xrange(0,N_mesh)]
        sep_vectors = [centres[i]-centres[n_cell] for i,n_cell in enumerate(neighbours)]
        norms = [np.linalg.norm(cell_vectors,axis=1) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_norms[:,np.newaxis],2,axis=1) for cell_norms,cell_vectors in zip(norms,sep_vectors)]
    
        return neighbours,norms,sep_vectors

class MeshTor(Mesh):
    
    def __init__(self,centres,width,height,N_mesh,save_areas=False):
        self.N_mesh = N_mesh
        self.width,self.height = width,height
        self.centres = centres
        if save_areas:
            self.get_neighbours = self.get_neighbours_with_areas
            self.update = self.update_with_areas
            self.neighbours, self.distances, self.unit_vecs, self.areas = self.get_neighbours()
        else:
            self.get_neighbours = self.get_neighbours_without_areas
            self.update = self.update_without_areas
            self.neighbours, self.distances, self.unit_vecs = self.get_neighbours()
            
    def copy(self):
        meshcopy = copy.copy(self)
        meshcopy.centres = copy.copy(meshcopy.centres)
        return meshcopy
    
    def update_with_areas(self):
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs, self.areas = self.get_neighbours()
    
    def update_without_areas(self):
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs = self.get_neighbours()
        
    def move(self, i, dr):
        coords = self.periodise(self.centres[i] + dr)
        self.centres[i] = coords
    
    def move_all(self, dr_array):
        self.centres = self.periodise_list(self.centres + dr_array)
    
    def periodise_list(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            coords[np.where(coords[:,i] >= L)[0],i] -= L*2
            coords[np.where(coords[:,i] < -L)[0],i] += L*2
        return coords
    
    def periodise(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            if coords[i] >= L: coords[i] -= L*2
            elif coords[i] < -L: coords[i] += L*2
        return coords
    
    def add(self,pos):
        self.periodise_list(pos)
        self.centres = np.append(self.centres,pos,0)
    
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
        return Voronoi(self.centres)
    
    def convex_hull(self):
        return ConvexHull(self.centres())  

    def delaunay(self,centres):
        return Delaunay(centres)
            
    def get_neighbours_without_areas(self):
        N_mesh = self.N_mesh
        centres,width,height = self.centres,self.width,self.height
        centres_3x3 = np.reshape([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]],(9*N_mesh,2))
        vnv = self.delaunay(centres_3x3).vertex_neighbor_vertices
        neighbours = [vnv[1][vnv[0][k]:vnv[0][k+1]] for k in xrange(4*N_mesh,5*N_mesh)]
        sep_vectors = [centres[i]-centres_3x3[n_cell] for i,n_cell in enumerate(neighbours)]
        # norms = np.sqrt((sep_vectors*sep_vectors).sum(axis=1))
        norms = [np.linalg.norm(cell_vectors,axis=1) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_norms[:,np.newaxis],2,axis=1) for cell_norms,cell_vectors in zip(norms,sep_vectors)]
        neighbours = [n_set%N_mesh for n_set in neighbours] 
    
        return neighbours,norms,sep_vectors
        
    def get_neighbours_with_areas(self):
        N_mesh = self.N_mesh
        centres,width,height = self.centres,self.width,self.height
        centres_3x3 = np.reshape([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]],(9*N_mesh,2))
        vor = Voronoi(centres_3x3)
        pairs = vor.ridge_points
        neighbours = [pairs[loc[0],1-loc[1]] for loc in (np.where(pairs==k) for k in xrange(4*N_mesh,5*N_mesh))]
        sep_vectors = [centres[i]-centres_3x3[n_cell] for i,n_cell in enumerate(neighbours)]
        norms = [np.linalg.norm(cell_vectors,axis=1) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_norms[:,np.newaxis],2,axis=1) for cell_norms,cell_vectors in zip(norms,sep_vectors)]
        neighbours = [n_set%N_mesh for n_set in neighbours] 
        areas = np.abs([self.get_cell_area(vor.vertices[polygon]) for polygon in np.array(vor.regions)[vor.point_region]])
    
        return neighbours,norms,sep_vectors,areas

    def get_cell_area(self,points):
        n_p = len(points)
        return 0.5*sum(points[i][0]*points[(i+1)%n_p][1]-points[(i+1)%n_p][0]*points[i][1] for i in range(n_p))

class MeshCyl(Mesh):
    
    def __init__(self,centres,width,height,N_mesh,save_areas=True):
        self.N_mesh = N_mesh
        self.width,self.height = width,height
        self.centres = centres
        if save_areas:
            self.get_neighbours = self.get_neighbours_with_areas
            self.update = self.update_with_areas
            self.neighbours, self.distances, self.unit_vecs, self.areas = self.get_neighbours()
        else:
            self.get_neighbours = self.get_neighbours_without_areas
            self.update = self.update_without_areas
            self.neighbours, self.distances, self.unit_vecs = self.get_neighbours()
            
    def copy(self):
        meshcopy = copy.copy(self)
        meshcopy.centres = copy.copy(meshcopy.centres)
        return meshcopy
    
    def update_with_areas(self):
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs, self.areas = self.get_neighbours()
    
    def update_without_areas(self):
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs = self.get_neighbours()
        
    def move(self, i, dr):
        coords = self.periodise(self.centres[i] + dr)
        self.centres[i] = coords
    
    def move_all(self, dr_array):
        self.centres = self.periodise_list(self.centres + dr_array)
    
    def periodise_list(self,coords):
        half_width = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            coords[np.where(coords[:,i] >= L)[0]][:,i] -= L*2
            coords[np.where(coords[:,i] < -L)[0]][:,i] += L*2
        return coords
    
    def periodise(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            if coords[i] >= L: coords[i] -= L*2
            elif coords[i] < -L: coords[i] += L*2
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
        return Voronoi(self.centres)
    
    def convex_hull(self):
        return ConvexHull(self.centres())  
            
    def get_neighbours_without_areas(self):
        N_mesh = self.N_mesh
        centres,width,height = self.centres,self.width,self.height
        centres_3x3 = np.reshape([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]],(9*N_mesh,2))
        vnv = Delaunay(centres_3x3).vertex_neighbor_vertices
        neighbours = [vnv[1][vnv[0][k]:vnv[0][k+1]] for k in xrange(4*N_mesh,5*N_mesh)]
        sep_vectors = [centres[i]-centres_3x3[n_cell] for i,n_cell in enumerate(neighbours)]
        norms = [np.linalg.norm(cell_vectors,axis=1) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_norms[:,np.newaxis],2,axis=1) for cell_norms,cell_vectors in zip(norms,sep_vectors)]
        neighbours = [n_set%N_mesh for n_set in neighbours] 
    
        return neighbours,norms,sep_vectors
        
    def get_neighbours_with_areas(self):
        N_mesh = self.N_mesh
        centres,width,height = self.centres,self.width,self.height
        centres_3x3 = np.reshape([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]],(9*N_mesh,2))
        vor = Voronoi(centres_3x3)
        pairs = vor.ridge_points
        neighbours = [pairs[loc[0],1-loc[1]] for loc in (np.where(pairs==k) for k in xrange(4*N_mesh,5*N_mesh))]
        sep_vectors = [centres[i]-centres_3x3[n_cell] for i,n_cell in enumerate(neighbours)]
        norms = [np.linalg.norm(cell_vectors,axis=1) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_norms[:,np.newaxis],2,axis=1) for cell_norms,cell_vectors in zip(norms,sep_vectors)]
        neighbours = [n_set%N_mesh for n_set in neighbours] 
        areas = np.abs([self.get_cell_area(vor.vertices[polygon]) for polygon in np.array(vor.regions)[vor.point_region]])
    
        return neighbours,norms,sep_vectors,areas

    def get_cell_area(self,points):
        n_p = len(points)
        return 0.5*sum(points[i][0]*points[(i+1)%n_p][1]-points[(i+1)%n_p][0]*points[i][1] for i in range(n_p))
                
    
if __name__ == '__main__':
    rand = np.random.RandomState(123456)
    mesh = MeshTor(rand.rand(100,2),1.0,1.0,100,False)
    

