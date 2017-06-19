import numpy as np
from mesh import Mesh

def init_centres(N_cell_across,N_cell_up,noise,rand):
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

def hex_mesh(N_across,N_up,noise,rand):
    return Mesh(init_centres(N_across,N_up,noise,rand))