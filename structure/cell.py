import numpy as np
import copy
from functools import partial
from global_constants import *

                
class Tissue(object):    
    
    def __init__(self,mesh,cell_ids=None,next_id=None,age=None,mother=None,properties=None):
        self.mesh = mesh
        N = len(mesh)
        self.cell_ids = cell_ids
        self.next_id = next_id
        self.age = age
        self.mother = mother
        self.properties = properties or {}
        
    def __len__(self):
        return len(self.mesh)
    
    def copy(self):
        return Tissue(self.mesh.copy(),self.cell_ids.copy(),self.next_id,self.age.copy(),self.mother.copy(),self.properties.copy())
    
    def move_all(self,dr_array):
        for i, dr in enumerate(dr_array):
            self.mesh.move(i,dr)
            
    def update(self,dt):
        self.mesh.update()
        self.age += dt
    
    def remove(self,idx_list):
        self.mesh.remove(idx_list)
        self.cell_ids = np.delete(self.cell_ids,idx_list)
        self.age = np.delete(self.age,idx_list)
        self.mother = np.delete(self.mother,idx_list)
        for key,val in self.properties.iteritems():
            self.properties[key] = np.delete(val,idx_list)
        
    def add_daughter_cells(self,i,rand):
        angle = rand.rand()*np.pi
        dr = np.array((EPS*np.cos(angle),EPS*np.sin(angle)))
        new_cen1 = self.mesh.centres[i] + dr
        new_cen2 = self.mesh.centres[i] - dr
        self.mesh.add([new_cen1,new_cen2])
        self.cell_ids = np.append(self.cell_ids,[self.next_id,self.next_id+1])
        self.age = np.append(self.age,[0.0,0.0])
        self.mother = np.append(self.mother,[self.cell_ids[i]]*2)
        self.next_id += 1

        
    def force_i(self,i,distances,vecs,n_list):
        # pref_sep = [self.age[i]*(L0-2*EPS) +2*EPS if self.mother[i] == self.mother[j] and self.age[i] <1.0
        #                 else L0 for j in n_list]
        # import ipdb; ipdb.set_trace()
        pref_sep = RHO+0.5*ALPHA*(self.age[n_list]+self.age[i])
        pref_sep[pref_sep>2.] = 2.
        return (MU*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)

    def dr(self,dt):
        distances,vecs,neighbours = self.mesh.distances,self.mesh.unit_vecs,self.mesh.neighbours
        return (dt/ETA)*np.array([self.force_i(i,dist,vec,neigh) for i,(dist,vec,neigh) in enumerate(zip(distances,vecs,neighbours))])
