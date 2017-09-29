import numpy as np
import copy
from functools import partial
from global_constants import *

class Cell(object):
    
    def __init__(self,rand,id,mother=None,age=0.,cycle_len=None):
        self.id = id
        if cycle_len is None: self.cycle_len = self.delayed_uniform(rand)
        else: self.cycle_len = cycle_len
        if mother is None: mother = id
        # self.aod = rand.exponential(T_G1+T_other) #age of death
        self.aod = self.delayed_uniform(rand)
        self.mother = mother
        self.age = age
    
    def copy(self):
        return Cell(self.id,self.mother,self.age,self.cycle_len)

    def clone(self,cell,new_id,rand):
        return Cell(rand,new_id,cell.id)
        
    def delayed_normal(self,rand,type=None):
        if type is None:
            G1_len = rand.normal(T_G1,np.sqrt(V_G1))
            G1_len = max(G1_len,min_G1)
        return G1_len + T_other
    
    def delayed_uniform(self,rand,type=None):
        if type is None:
            G1_len = rand.rand()*T_G1*2
        return G1_len + T_other
    
    def poisson_death(self,rand):
        return rand.exponential(T_other+T_G1)
            
            
                
class Tissue(object):    
    
    def __init__(self,mesh,cell_array,next_id):
        self.mesh = mesh
        self.cell_array = np.array(cell_array)
        self.next_id = next_id
        
    def __len__(self):
        return len(self.mesh)
    
    def by_mesh(self,key):
        return np.array([cell.__dict__[key] for cell in self.cell_array])
    
    def by_mesh_ids(self,key,ids):
        return np.array([cell.__dict__[key] for cell in self.cell_array[ids]])
        
    def set_attributes(self,key,list_like):
        for cell,attr in zip(self.cell_array,list_like):
            cell.__dict__[key] = attr         
    
    def ready(self):
        return [i for i,cell in enumerate(self.cell_array) if cell.age >= cell.cycle_len]
    
    def dead(self):
        return [i for i,cell in enumerate(self.cell_array) if cell.age >= cell.aod]
    
    def copy(self):
        return Tissue(self.mesh.copy(),copy.deepcopy(self.cell_array),self.next_id)
    
    def move_all(self,dr_array):
        for i, dr in enumerate(dr_array):
            self.mesh.move(i,dr)
    
    def remove(self,idx_list):
        self.mesh.remove(idx_list)
        self.cell_array = np.delete(self.cell_array,idx_list)
    
    def add_clone(self,cell,pos,rand):
        self.mesh.add(pos)
        self.cell_array = np.append(self.cell_array,cell.clone(cell,self.next_id,rand))
        self.next_id += 1
        
    def cell_division(self,i,rand):
        cell = self.cell_array[i]
        angle = rand.rand()*np.pi
        dr = np.array((EPS*np.cos(angle),EPS*np.sin(angle)))
        new_cen1 = self.mesh.centres[i] + dr
        new_cen2 = self.mesh.centres[i] - dr
        self.add_clone(cell,new_cen1,rand)
        self.add_clone(cell,new_cen2,rand)
        self.remove(i)
    
    def update(self,dt):
        self.mesh.update()
        for cell in self.cell_array:
            cell.age += dt
        
    def pref_sep(self,i,j):
        if self.cell_array[i].mother == self.cell_array[j].mother:
            age_i = self.cell_array[i].age
            if age_i < 1.0:
                return age_i*(L0-2*EPS) +2*EPS
        return L0

    def force_ij(self,i,j):      
        if r_len > r_max: return np.array((0.0,0.0))
        else: return MU*r_hat*(r_len-self.pref_sep(i,j))

    def force_i(self,i,distances,vecs,n_list):
        cell = self.cell_array[i]
        pref_sep = [cell.age*(L0-2*EPS) +2*EPS if cell.mother == self.cell_array[j].mother and cell.age <1.0
                        else L0 for j in n_list]
        return sum(MU*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1))
        
    def force_adhesion_i(self,i,distances,vecs,n_list):
        XI = 0.1
        cell = self.cell_array[i]
        pref_sep = L0
        MU_vec = np.full(len(n_list),MU)
        mask = (distances>L0)*(self.by_mesh_ids('type',n_list)!=cell.type)
        MU_vec[mask] *= 0.9
        MU_vec[distances>r_max] = 0
        pert = np.sqrt(2*XI/dt)*np.random.normal(0,1,2)
        return sum(np.repeat((MU_vec)[:,np.newaxis],2,axis=1)*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)) + pert
    
    def dr(self,dt):
        distances,vecs,neighbours = self.mesh.distances,self.mesh.unit_vecs,self.mesh.neighbours
        return (dt/ETA)*np.array([self.force_i(i,dist,vec,neigh) for i,(dist,vec,neigh) in enumerate(zip(distances,vecs,neighbours))])
