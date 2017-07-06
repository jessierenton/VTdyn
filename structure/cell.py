class Cell(object):
    
    def init(self,id,meshpt,ghost=False,**kwargs):
        self.id = id
        self.meshpt = meshpt
        self.ghost = ghost
        for key, value in kwargs.items():
            setattr(self, key, value)

class Tissue(object):    
    def init(self,mesh,cell_array):
        self.mesh = mesh
        self.cell_array = cell_array
        
    def cell_division(self,cell_pos):
        cell = self.cell_array(cell_pos)
        self.cell_array.pop(cell_pos)
        angle = rand.rand()*np.pi
        dr = np.array((EPS*np.cos(angle),EPS*np.sin(angle)))
        new_cen1 = self.mesh.centres[idx] + dr
        new_cen2 = self.mesh.centres[idx] - dr
        self.mesh.add(new_cen1)
        self.mesh.add(new_cen2)
        self.mesh.remove(cell.meshpt)