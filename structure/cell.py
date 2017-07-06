import numpy as numpy
import copy

class Cell(object):
    
    def __init__(self,id,mesh_id,mother,age=0.,cycle_len=None):
        self.id = id
        self.mesh_id = mesh_id
        if cycle_len is None: self.cycle_len = self.cycle_dist()
        else: self.cycle_len = cycle_len
        self.parent = parent
        self.age = age

    def clone(self,id,mesh_id,mother):
        return Cell(id,mesh_id,mother)
        
    def cycle_dist(self,type=None,rand=rand):
        if type is None:
            cycle_len = rand.normal(T_G1,np.sqrt(V_G1))
            lifetimes[np.where(lifetimes<min_G1)[0]]==min_G1
        return lifetimes + T_other
            
    
class Tissue(object):    
    
    def __init__(self,mesh,cell_array,next_id):
        self.mesh = mesh
        self.cell_array = np.array(cell_array)
        self.next_id = next_id
        
    def cell_division(self,cell_pos):
        cell = self.cell_array(cell_pos)
        self.cell_array = np.delete(self.cell_array,cell_pos)
        angle = rand.rand()*np.pi
        dr = np.array((EPS*np.cos(angle),EPS*np.sin(angle)))
        mother_centre = centre(cell.mesh_id)
        new_cen1 = mother_centre + dr
        new_cen2 = mother_centre - dr
        mesh_id1 = self.mesh.add(new_cen1)
        mesh_id2 = self.mesh.add(new_cen2)
        self.mesh.remove(cell.mesh_id)
        new_cell1 = cell.clone(self.next_id,mesh_id1,cell.id)
        new_cell2 = cell.clone(self.next_id+1,mesh_id2,cell.id)
        self.next_id += 2
    
    def cell_apoptosis(self,cell_pos):
        cell = self.cell_array(cell_pos)
        self.mesh.remove(cell.mesh_id)
    
    def update(self,dt):
        self.mesh.update()
        self.cell_array.
        