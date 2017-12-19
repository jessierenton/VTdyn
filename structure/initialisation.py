import numpy as np
from cell import Tissue
from mesh import Mesh,Torus

def hex_centres(N_across,N_up,noise,rand,multiplier=1):
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
    
    return centres*multiplier, width*multiplier, height*multiplier

def init_mesh(N_cell_across,N_cell_up,noise,rand,mutant=None,save_areas=False):
    centres= hex_centres(N_cell_across,N_cell_up,noise,rand)[0]
    mesh = Mesh(centres)
    return mesh
    
def init_mesh_torus(N_cell_across,N_cell_up,noise,rand,multiplier=1,mutant=None,save_areas=False):
    centres,width,height = hex_centres(N_cell_across,N_cell_up,noise,rand,multiplier)
    return Mesh(centres,Torus(width,height))
    
def init_tissue_torus(N_cell_across,N_cell_up,noise,rand,mutant=None,save_areas=False):
    N = N_cell_across*N_cell_up
    return Tissue(init_mesh_torus(N_cell_across,N_cell_up,noise,rand,mutant,save_areas=save_areas),np.arange(N),
                N,np.zeros(N,dtype=float),np.full(N,-1,dtype=int))
    
def init_tissue_torus_with_multiplier(N_cell_across,N_cell_up,noise,rand,multiplier,ages=None,mutant=None,save_areas=False):
    if ages is None: ages = np.zeros(N,dtype=float)
    N = N_cell_across*N_cell_up
    return Tissue(init_mesh_torus(N_cell_across,N_cell_up,noise,rand,multiplier,mutant,save_areas=save_areas),np.arange(N),
                N,ages,np.full(N,-1,dtype=int))
    
    
    
    