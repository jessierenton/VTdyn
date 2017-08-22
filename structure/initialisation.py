import numpy as np
from cell import Cell, Tissue
from mesh import Mesh

def hex_centres(N_cell_across,N_cell_up,noise,rand):
    """generates positions of cell centres arranged (approximately) hexagonally 
    surrounded a border by ghost nodes
    """
    
    assert(N_cell_up % 2 == 0)  # expect even number of rows
    dx, dy = 1.0/N_cell_across, 1.0/(N_cell_up/2)
    x = np.arange(-0.5+dx/4, 0.5, dx)
    y = np.arange(-0.5+dy/4, 0.5, dy)
    centres = np.zeros((N_cell_across, N_cell_up/2, 2, 2))
    centres[:, :, 0, 0] += x[:, np.newaxis]
    centres[:, :, 0, 1] += y[np.newaxis, :]
    x += dx/2
    y += dy/2
    centres[:, :, 1, 0] += x[:, np.newaxis]
    centres[:, :, 1, 1] += y[np.newaxis, :]

    ratio = np.sqrt(2/np.sqrt(3))
    width = N_cell_across*ratio
    height = N_cell_up/ratio

    centres = centres.reshape(-1, 2)*np.array([width, height])
    centres += rand.rand(N_cell_up*N_cell_across, 2)*noise
  
    return centres

# def circular_random_mesh(N_cell, rand):
#     R = np.sqrt(N_cell/np.pi)
#     r = R*np.sqrt(rand.rand(N_cell))
#     theta = 2*np.pi*rand.rand(N_cell)
#     centres = np.array((r*np.cos(theta), r*np.sin(theta))).T
#     reflected_centres = (R*R/r/r)[:, None]*centres

def init_tissue(N_cell_across,N_cell_up,noise,rand):
    centres= hex_centres(N_cell_across,N_cell_up,noise,rand)
    mesh = Mesh(centres)
    cell_array = basic_cells(mesh,rand)
    return Tissue(mesh,cell_array,len(mesh))
    

def basic_cells(mesh,rand):
    cells = [Cell(rand,id,-1,age=1.,cycle_len=None)
        for id,centre in zip(range(len(mesh)),mesh.centres)]
    return cells
    

    
    
    
    
    
    
    