import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import seaborn as sns

sns.set_style("white")

#cell-cycle times hours
T_G1, T_other = 2,10
V_G1 = 1 #variance in G1-time
min_G1 = 0.01 #min G1-time

L0 = 1.0
EPS = 0.05

class Cells(object):
    
    def __init__(self,mesh,cell_ids=None,properties=None,rand=None,):
        self.mesh = mesh
        if properties is None:
            self.properties = {}
            self.properties['sister'] = np.full(self.mesh.N_tot,-1,dtype=int)
            self.properties['lifespan'] = rand.rand(self.mesh.N_tot)*10
            self.properties['age'] = np.full(self.mesh.N_tot,np.inf)
        else: self.properties = properties
        if cell_ids is None: self.cell_ids = np.arange(mesh.N_tot)
        else: self.cell_ids = cell_ids
        self.next_id = mesh.N_tot
    
    def id_to_idx(self,id):
        return np.where(self.cell_ids == id)[0][0]
        
    def idx_to_id(idx):
        return self.cell_ids[idx]
        
    def copy(self):
        return Cells(self.mesh.copy(),self.cell_ids.copy(),self.properties.copy())
        
    def __len__(self):
        return self.mesh.N_cells
    
    def cell_division(self,mother,rand):
        idx = self.id_to_idx(mother)
        angle = rand.rand()*np.pi
        dr = np.array((EPS*np.cos(angle),EPS*np.sin(angle)))
        centre1 = self.mesh.centres[idx] + dr
        centre2 = self.mesh.centres[idx] - dr
        gm = self.mesh.ghost_mask[idx]
        self.mesh.add(centre1,gm)
        self.mesh.add(centre2,gm)
        self.cell_ids = np.delete(self.cell_ids,idx)
        self.mesh.remove(idx)
        self.cell_ids = np.append(self.cell_ids,(self.next_id,self.next_id+1))
        self.properties['sister'] = np.append(self.properties['sister'],[self.next_id+1,self.next_id])
        self.next_id += 2
        self.properties['age'] = np.append(self.properties['age'],[0.0,0.0])
        self.properties['lifespan'] = np.append(self.properties['lifespan'],assign_lifetime(2,rand))
        
    def cell_apoptosis(self,dead):
        idx = id_to_idx(dead)
        self.cell_ids = np.delete(self.cell_ids,idx)
        self.mesh.remove(idx)
    
    def pref_sep(self,i,j):
        age_i = self.properties['age'][self.cell_ids[i]]
        if age_i < 1.0 and self.properties['age'][self.cell_ids[i]] == j:
            return age_i*(L0-2*EPS) +2*EPS
        else: return L0
    
    def update(self,dt):
        self.mesh.update()
        self.properties['lifespan'] -= dt
        self.properties['age'] -= dt
    
    def plot_cells(self,label=False):
        fig = plt.Figure()        
        ax = plt.axes()
        plt.axis('scaled')
        xmin,xmax = min(self.mesh.centres[:,0]), max(self.mesh.centres[:,0])
        ymin,ymax = min(self.mesh.centres[:,1]), max(self.mesh.centres[:,1])      
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        vor = self.mesh.voronoi()
        cells_by_vertex = np.array(vor.regions)[np.array(vor.point_region)]
        verts = [vor.vertices[cv] for cv in cells_by_vertex[self.mesh.ghost_mask]]
        coll = PolyCollection(verts,linewidths=[2.])
        ax.add_collection(coll)
        if label:
            for i, coords in enumerate(self.mesh.centres):
                if self.mesh.ghost_mask[i]: plt.text(coords[0],coords[1],str(self.cell_ids[i]))
                
        plt.show()

def assign_lifetime(num_cells,rand):
    lifetimes = rand.normal(T_G1,np.sqrt(V_G1),num_cells)
    lifetimes[np.where(lifetimes<min_G1)[0]]==min_G1
    return lifetimes + T_other
    