import numpy as np
import copy
from functools import partial
import global_constants as gc
from global_constants import EPS, L0, MU, ETA
              
class Tissue(object):    
    
    """Defines a tissue comprised of cells which can move, divide and be extruded"""
    
    def __init__(self,mesh,force,cell_ids,next_id,age,mother,properties=None,extruded_cells=None,divided_cells=None,store_dead=False,time=0.):
        """ Parameters:
        mesh: Mesh object
            defines cell locations and neighbour connections
        force: Force object
            defines force law between neighbouring cells
        cell_ids: (N,) array ints
             unique id for each cell (N is number of cells)
        next_id: int
            next available cell id
        age: (N,) array floats
            age of each cell
        mother: (N,) array ints
            id of mother for each cell (-1 for initial cells)
        properties: dict or None
            dictionary available for any other cell properties
            
        """
        self.mesh = mesh
        self.Force = force
        self.cell_ids = cell_ids
        self.next_id = next_id
        self.age = age
        self.mother = mother
        self.properties = properties or {}
        self.store_dead = store_dead
        if store_dead:    
            self.extruded_cells = extruded_cells or []
            self.divided_cells = divided_cells or []
        self.time=time
        
        
    def __len__(self):
        return len(self.mesh)
    
    def reset(self,reset_age=True):
        N = len(self)
        self.cell_ids = np.arange(N,dtype=int)
        if reset_age: self.age = np.zeros(N,dtype=float)
        self.next_id = N
        self.mother = -np.ones(N,dtype=int)
        self.time = 0.
        
    
    def copy(self):
        """create a copy of Tissue"""
        if self.store_dead:
            return Tissue(self.mesh.copy(),self.Force,self.cell_ids.copy(),self.next_id,self.age.copy(), self.mother.copy(),copy.deepcopy(self.properties),self.extruded_cells[:],self.divided_cells[:], self.store_dead,self.time)
        else: return Tissue(self.mesh.copy(),self.Force,self.cell_ids.copy(),self.next_id,self.age.copy(), self.mother.copy(),copy.deepcopy(self.properties),time=self.time)
    
	def mesh_id(self,cell_id):
		return np.where(self.mesh.ids==cell_id)[0]
        
    def update(self,dt):
        self.mesh.update()
        self.age += dt      
        self.time += dt
    
    def update_extruded_divided_lists(self,idx_list,mother):
        if isinstance(idx_list,int):
            if mother:
                self.divided_cells.append((self.cell_ids[idx_list],self.age[idx_list],self.time))
            else:
                self.extruded_cells.append((self.cell_ids[idx_list],self.age[idx_list],self.time))  
        else:
            if mother is True:
                divided_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list],self.age[idx_list])]
                self.divided_cells.extend(divided_cells)
            elif mother is False: 
                extruded_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list],self.age[idx_list])]
                self.extruded_cells.extend(extruded_cells)    
            else:
                divided_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list[mother]],self.age[idx_list[mother]])]
                extruded_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list[~mother]],self.age[idx_list[~mother]])]
                self.divided_cells.extend(divided_cells)
                self.extruded_cells.extend(extruded_cells)
    
    def remove(self,idx_list,mother=None):
        """remove a cell (or cells) from tissue. if storing dead cell ids need arg mother=True if cell is being removed
        following division, false otherwise. can be list."""
        if self.store_dead:
             self.update_extruded_divided_lists(idx_list,mother)
        self.mesh.remove(idx_list)
        self.cell_ids = np.delete(self.cell_ids,idx_list)
        self.age = np.delete(self.age,idx_list)
        self.mother = np.delete(self.mother,idx_list)
        for key,val in self.properties.iteritems():
            self.properties[key] = np.delete(val,idx_list)
        
    def add_daughter_cells(self,i,rand,daughter_properties=None):
        """add pair of new cells after a cell division. copies properties dictionary from mother unless alternative values
        are specified in the daughter_properties argument"""
        angle = rand.rand()*np.pi
        dr = np.array((EPS*np.cos(angle),EPS*np.sin(angle)))
        new_cen1 = self.mesh.centres[i] + dr
        new_cen2 = self.mesh.centres[i] - dr
        self.mesh.add([new_cen1,new_cen2])
        self.cell_ids = np.append(self.cell_ids,[self.next_id,self.next_id+1])
        self.age = np.append(self.age,[0.0,0.0])
        self.mother = np.append(self.mother,[self.cell_ids[i]]*2)
        self.next_id += 2
        for key,val in self.properties.iteritems():
            if daughter_properties is None or key not in daughter_properties: 
                self.properties[key] = np.append(self.properties[key],[self.properties[key][i]]*2)
            else: 
                self.properties[key] = np.append(self.properties[key],daughter_properties[key])  
    
    def add_many_daughter_cells(self,idx_list,rand):
         if len(idx_list)==1: self.add_daughter_cells(idx_list[0],rand)
         else:
             for i in idx_list:
                 self.add_daughter_cells(i,rand)
        
    def dr(self,dt): 
        """calculate distance cells move due to force law in time dt"""  
        return (dt/ETA)*self.Force(self)
    
    def cell_stress(self,i):
        """calculates the stress p_i on a single cell i according to the formula p_i = sum_j mag(F^rep_ij.u_ij)/l_ij
        where F^rep_ij is the repulsive force between i and j, i.e. F^rep_ij=F_ij if Fij is positive, 0 otherwise;
        u_ij is the unit vector between the i and j cell centres and l_ij is the length of the edge between cells i and j"""     
        edge_lengths = self.mesh.edge_lengths(i)
        repulsive_forces = self.Force.force_ij(self,i)
        repulsive_forces[repulsive_forces<0]=0
        return sum(repulsive_forces/edge_lengths) 
    
    def tension_area_product(self,i):
        distances = self.mesh.distances[i]
        forces = self.Force.force_ij(self,i)
        return -0.25*sum(forces*distances)
        
class Force(object):
    """Abstract force object"""
    def force(self):
        """returns (N,2) array floats giving vector force on each cell"""
        raise NotImplementedError()
    
    def magnitude(self,tissue):
        """returns (N,) array floats giving magnitude of force on each cell"""
        return np.sqrt(np.sum(np.sum(self.force(tissue),axis=0)**2))
    
    def __call__(self, tissue):
        return self.force(tissue)
        
class BasicSpringForceTemp(Force):
    
    def __init__(self,MU=MU):
        self.MU=MU
    
    def force(self,tissue):
        return np.array([self.force_i(tissue,i) for i in range(len(tissue))])
    
    def force_i(self):
        """returns force on cell i"""
        raise Exception('force law undefined')

class BasicSpringForceNoGrowth(BasicSpringForceTemp):
    
    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        if tissue.age[i] >= 1.0 or tissue.mother[i] == -1: pref_sep = L0
        else: pref_sep = (tissue.mother[n_list]==tissue.mother[i])*((L0-EPS)*tissue.age[i]+EPS-L0) +L0
        return (self.MU*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)
    
    def force_ij(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        if tissue.age[i] >= 1.0 or tissue.mother[i] == -1: pref_sep = L0
        else: pref_sep = (tissue.mother[n_list]==tissue.mother[i])*((L0-EPS)*tissue.age[i]+EPS-L0) +L0
        forces = self.MU*(distances-pref_sep) 
        return forces

class BasicSpringForceGrowth(BasicSpringForceTemp):

    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        return (self.MU*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)

class SpringForceVariableMu(BasicSpringForceTemp):

    def __init__(self,delta,MU=MU):
        BasicSpringForceTemp.__init__(MU)
        self.delta = delta

    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        MU_list = self.MU*(1-0.5*self.delta*(tissue.properties['mutant'][n_list]+tissue.properties['mutant'][i]))
        return (vecs*np.repeat((MU_list*(distances-pref_sep))[:,np.newaxis],2,axis=1)).sum(axis=0)

class MutantSpringForce(BasicSpringForceTemp):

    def __init__(self,alpha,MU=MU):
        BasicSpringForceTemp.__init__(MU)
        self.alpha = alpha

    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        alpha_i = tissue.properties['mutant'][i]*(self.alpha-1)+1
        return (vecs*np.repeat((MU/alpha_i*(distances-pref_sep))[:,np.newaxis],2,axis=1)).sum(axis=0)
        
