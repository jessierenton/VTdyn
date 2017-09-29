import numpy as np
from cell import Cell, Tissue
from mesh import Mesh,MeshTor

def hex_centres(N_across,N_up,noise,rand):
    assert(N_up % 2 == 0)  # expect even number of rows    
    width, height = float(N_across), float(N_up)*np.sqrt(3)/2
    x = np.arange(-width/2.,width/2.,width/N_across)
    y = np.arange(-height/2.,height/2.,height/(N_up/2))
    centres = np.zeros((N_across, N_up/2, 2, 2))
    centres[:, :, 0, 0] += x[:, np.newaxis]
    centres[:, :, 1, 0] += (x+1./2)[:, np.newaxis]
    centres[:, :, 0, 1] += y[np.newaxis, :]
    centres[:, :, 1, 1] += (y+np.sqrt(3)/2)[np.newaxis,:]
    centres = centres.reshape(-1, 2) + np.array([0.25,3**0.5/4])
    centres += (rand.rand(N_up*N_across, 2)-0.5)*noise 
    
    return centres, width, height

def init_tissue(N_cell_across,N_cell_up,noise,rand,mutant=None):
    centres= hex_centres(N_cell_across,N_cell_up,noise,rand)[0]
    mesh = Mesh(centres)
    cell_array = basic_cells(mesh,rand,ptype=0)
    if mutant: cell_array[rand.randint(N_cell_across*N_cell_up,size=mutant)].ptype = 1
    return Tissue(mesh,cell_array,len(mesh))
    
def init_tissue_torus(N_cell_across,N_cell_up,noise,rand,mutant=None):
    centres,width,height = hex_centres(N_cell_across,N_cell_up,noise,rand)
    mesh = MeshTor(centres,width,height,N_cell_across*N_cell_up)
    cell_array = basic_cells(mesh,rand,ptype=0)
    return Tissue(mesh,cell_array,len(mesh))

def basic_cells(mesh,rand,ptype=None):
    cells = [Cell(rand,id,age=1.,cycle_len=None)
        for id,centre in zip(range(len(mesh)),mesh.centres)]
    return cells
    
    
    

    
    
    
    
    
    
    