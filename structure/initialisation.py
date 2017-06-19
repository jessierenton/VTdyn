import numpy as np

def init_centres(N_cell_across,N_cell_up,ghost_num,noise,rand):
    """generates positions of cell centres arranged (approximately) hexagonally 
    surrounded a border by ghost nodes
    """
    N_cell_across += 2*ghost_num
    N_cell_up += 2*ghost_num
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
    
    ghost_mask = np.full(N_cell_across*N_cell_up,True, dtype=bool)
    ghost_mask[:N_cell_up*ghost_num] = False
    ghost_mask[(N_cell_across-ghost_num)*N_cell_up:] = False
    for i in range(1,N_cell_across):
        ghost_mask[N_cell_up*i-ghost_num:N_cell_up*i+ghost_num] = False 
    return centres, ghost_mask