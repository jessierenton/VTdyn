import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull
import copy

def polygon_area(points):
    n_p = len(points)
    return 0.5*sum(points[i][0]*points[(i+1)%n_p][1]-points[(i+1)%n_p][0]*points[i][1] for i in range(n_p))

class Geometry(object):
    def periodise(self,r):
        return r
    
    def periodise_list(self,r):
        return r 
        
class Plane(Geometry):
    pass
    
    def retriangulate(self,centres,N_mesh):
        vor = Voronoi(centres)
        pairs = vor.ridge_points
        neighbours = [pairs[loc[0],1-loc[1]] for loc in (np.where(pairs==k) for k in xrange(N_mesh))]
        sep_vectors = [centres[i]-centres[n_cell] for i,n_cell in enumerate(neighbours)]
        norms = [np.sqrt((cell_vectors*cell_vectors).sum(axis=1)) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_norms[:,np.newaxis],2,axis=1) for cell_norms,cell_vectors in zip(norms,sep_vectors)]
        neighbours = [n_set for n_set in neighbours] 
        areas = np.abs([polygon_area(vor.vertices[polygon]) for polygon in np.array(vor.regions)[vor.point_region]])
        return neighbours, norms, sep_vector, areas

class Torus(Geometry):
    def __init__(self,width,height):
        self.width = width
        self.height = height
        
    def periodise(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            if coords[i] >= L: coords[i] -= L*2
            elif coords[i] < -L: coords[i] += L*2
        return coords
        
    def periodise_list(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            coords[np.where(coords[:,i] >= L)[0],i] -= L*2
            coords[np.where(coords[:,i] < -L)[0],i] += L*2
        return coords

    def retriangulate(self,centres,N_mesh):
        width,height = self.width, self.height
        centres_3x3 = np.reshape([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]],(9*N_mesh,2))
        vor = Voronoi(centres_3x3)
        pairs = vor.ridge_points
        neighbours = [pairs[loc[0],1-loc[1]] for loc in (np.where(pairs==k) for k in xrange(4*N_mesh,5*N_mesh))]
        sep_vectors = [centres[i]-centres_3x3[n_cell] for i,n_cell in enumerate(neighbours)]
        norms = [np.sqrt((cell_vectors*cell_vectors).sum(axis=1)) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_norms[:,np.newaxis],2,axis=1) for cell_norms,cell_vectors in zip(norms,sep_vectors)]
        neighbours = [n_set%N_mesh for n_set in neighbours] 
        areas = np.abs([polygon_area(vor.vertices[polygon]) for polygon in np.array(vor.regions)[vor.point_region][4*N_mesh:5*N_mesh]])
        return neighbours, norms, sep_vectors, areas

# class Cylinder(Geometry):
#     def __init__(self,width):
#         self.width = width
#
#     def periodise(self,coords):
#         half_width = self.width/2.
#         if coords[0] >= half_width: coords[i] -= half_width*2
#         elif coords[0] < -half_width: coords[i] += half_width*2
#         return coords
#
#     def periodise_list(self,coords):
#         half_width = self.width/2.
#         coords[np.where(coords[:,0] >= half_width)[0],i] -= half_width*2
#         coords[np.where(coords[:,0] < -half_width)[0],i] += half_width*2
#         return coords


class Mesh(object):
    """ 
    Attributes: N_cells = number of cells 
                centres = array of (x,y) values for both cell and ghost node positions
                vnv = contains neighbour information for cells (extract using neighbours method)
                tri = (N_mesh,3) array containing indices of centres forming triangles (used for plot_tri method)
    """
    
    def __init__(self,centres,geometry):
        self.N_mesh = len(centres)
        self.centres = centres
        self.geometry = geometry
        self.neighbours,self.distances,self.unit_vecs, self.areas = self.retriangulate()
    
    def __len__(self):
        return self.N_mesh
    
    def copy(self):
        meshcopy = copy.copy(self)
        meshcopy.centres = copy.copy(meshcopy.centres)
        return meshcopy
    
    def update(self):
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs, self.areas = self.retriangulate()
    
    def extreme_point(self):
        return np.max(self.distances())
    
    def distances(self):
        return np.array([np.sqrt(r[0]**2+r[1]**2) for r in self.centres])
        
    def move(self, i, dr):
        self.centres[i] = self.geometry.periodise(self.centres[i]+dr)
    
    def move_all(self, dr_array):
        self.centres = self.geometry.periodise_list(self.centres + dr_array)
    
    def add(self,pos):
        self.centres = np.append(self.centres,pos,0)
    
    def remove(self,i):
        self.centres = np.delete(self.centres,i,0)
        
    def voronoi(self):
        return Voronoi(self.centres)
    
    def convex_hull(self):
        return ConvexHull(self.centres())  

    def delaunay(self,centres):
        return Delaunay(centres)
    
    def retriangulate(self):
        N_mesh = self.N_mesh
        return self.geometry.retriangulate(self.centres,N_mesh)
    
