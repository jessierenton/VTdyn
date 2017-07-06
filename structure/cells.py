import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
from scipy.spatial import voronoi_plot_2d
import seaborn as sns

sns.set_style("white")


#cell-cycle times hours
T_G1, T_other = 2,10
V_G1 = 1 #variance in G1-time
min_G1 = 0.01 #min G1-time

L0 = 1.0
EPS = 0.05

class Cells(object):
    
    def __init__(self,mesh,cell_ids=None,properties={},rand=None,):
        self.mesh = mesh
        self.properties = properties
        if cell_ids is None: self.mesh.cell_ids = np.arange(mesh.N_tot)
        else: self.mesh.cell_ids = mesh.cell_ids
        self.next_id = mesh.N_tot
    
    def by_meshidx(self,property,ghosts=True):
        if ghosts: return self.properties[property][self.mesh.cell_ids]
        else: return self.properties[property][self.mesh.cell_ids][self.mesh.ghost_mask]
    
    def id_to_idx(self,id):
        return np.where(self.mesh.cell_ids == id)[0][0]
        
    def idx_to_id(self,idx):
        return self.mesh.cell_ids[idx]
        
    def copy(self):
        return Cells(self.mesh.copy(),self.mesh.cell_ids.copy(),self.properties.copy())
        
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
        self.mesh.cell_ids = np.delete(self.mesh.cell_ids,idx)
        self.mesh.remove(idx)
        self.mesh.cell_ids = np.append(self.mesh.cell_ids,(self.next_id,self.next_id+1))
        self.properties['sister'] = np.append(self.properties['sister'],[self.next_id+1,self.next_id])
        self.next_id += 2
        self.properties['age'] = np.append(self.properties['age'],[0.0,0.0])
        self.properties['lifespan'] = np.append(self.properties['lifespan'],assign_lifetime(2,rand))
        self.properties['deathtime'] = np.append(self.properties['deathtime'],assign_deathtime(2,rand))
        for key in self.properties:
            if key != 'age' and key != 'lifespan' and key != 'deathtime' and key != 'sister':
                self.properties[key] = np.append(self.properties[key],[self.properties[key][mother]]*2)
        
    def cell_apoptosis(self,dead):
        idx = self.id_to_idx(dead)
        self.mesh.cell_ids = np.delete(self.mesh.cell_ids,idx)
        self.mesh.remove(idx)
    
    def pref_sep(self,i,j):
        age_i = self.properties['age'][self.mesh.cell_ids[i]]
        if age_i < 1.0 and self.properties['sister'][self.mesh.cell_ids[i]] == self.mesh.cell_ids[j]:
            return age_i*(L0-2*EPS) +2*EPS
        else: return L0
    
    def update(self,dt):
        self.mesh.update()
        self.properties['deathtime'] -= dt
        self.properties['lifespan'] -= dt
        self.properties['age'] += dt
    
    def plot_basic(self):
        vor = self.mesh.voronoi()
        voronoi_plot_2d(vor)
        plt.show()
    

def assign_lifetime(num_cells,rand):
    lifetimes = rand.normal(T_G1,np.sqrt(V_G1),num_cells)
    lifetimes[np.where(lifetimes<min_G1)[0]]==min_G1
    return lifetimes + T_other
    
def assign_deathtime(num_cells,rand):
    return 12. + rand.exponential(1,num_cells)
    