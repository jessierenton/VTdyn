import numpy as np
import copy
from functools import partial
from global_constants import *
              
class Tissue(object):    

    def __init__(self,mesh,force,cell_ids=None,next_id=None,age=None,mother=None,properties=None):
        self.mesh = mesh
        self.Force = force
        N = len(mesh)
        self.cell_ids = cell_ids
        self.next_id = next_id
        self.age = age
        self.mother = mother
        self.properties = properties or {}
        
    def __len__(self):
        return len(self.mesh)
    
    def copy(self):
        return Tissue(self.mesh.copy(),self.Force,self.cell_ids.copy(),self.next_id,self.age.copy(),self.mother.copy(),self.properties.copy())
    
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
        self.next_id += 2
        
    def dr(self,dt):   
        return (dt/ETA)*self.Force(self)
        
        
class Force(object):
    
    def force(self):
        raise Exception('force undefined')
    
    def magnitude(self,tissue):
        return np.sqrt(np.sum(np.sum(self.force(tissue),axis=0)**2))
    
    def __call__(self, tissue):
        return self.force(tissue)
        
class BasicSpringForceTemp(Force):
    
    def force(self,tissue):
        return np.array([self.force_i(tissue,i,dist,vec,neigh) for i,(dist,vec,neigh) in enumerate(zip(tissue.mesh.distances,tissue.mesh.unit_vecs,tissue.mesh.neighbours))])
    
    def force_i(self):
        raise Exception('force law undefined')
    

class BasicSpringForceGrowth(BasicSpringForceTemp):
    
       def force_i(self,tissue,i,distances,vecs,n_list):
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        return (MU*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)

class BasicSpringForceNoGrowth(BasicSpringForceTemp):
    
    def force_i(self,tissue,i,distances,vecs,n_list):
        if tissue.age[i] >= 1.0 or tissue.mother[i] == -1: pref_sep = L0
        else: pref_sep = (tissue.mother[n_list]==tissue.mother[i])*((L0-EPS)*tissue.age[i]+EPS-L0) +L0
        return (MU*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)    
    
class SpringForceVariableMu(BasicSpringForceTemp):

    def __init__(self,delta):
        self.delta = delta 

    def force_i(self,tissue,i,distances,vecs,n_list):
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        MU_list = MU*(1-0.5*self.delta*(tissue.properties['mutant'][n_list]+tissue.properties['mutant'][i]))
        return (vecs*np.repeat((MU_list*(distances-pref_sep))[:,np.newaxis],2,axis=1)).sum(axis=0)

class MutantSpringForce(BasicSpringForceTemp):
    
    def __init__(self,alpha):
        self.alpha = alpha
        
    def force_i(self,tissue,i,distances,vecs,n_list):
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        alpha_i = tissue.properties['mutant'][i]*(self.alpha-1)+1
        return (vecs*np.repeat((MU/alpha_i*(distances-pref_sep))[:,np.newaxis],2,axis=1)).sum(axis=0)
        