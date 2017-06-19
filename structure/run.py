import sys
import numpy as np
import itertools
import mesh
from mesh import Mesh
from cells import Cells
from functools import partial
from initialisation import *

MU = -50.
ETA = 1.0
dt = 0.005 #hours


rand = np.random.RandomState(123456)

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.4f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def force_ij(mesh,i,j):
    if (mesh.ghost_mask[i] and not mesh.ghost_mask[j]): return np.array((0.0,0.0))
    else:              
        r_len, r_hat = mesh.seperation(i,j)
        return MU*r_hat*(r_len-cells.pref_sep(i,j))

def force_i(mesh,i):
    mapfunc = partial(force_ij,mesh,i)
    return sum(map(mapfunc,mesh.neighbours(i)))
    
def dr(mesh,dt):
    return (dt/ETA)*np.array([force_i(mesh,i) for i in range(0,mesh.N_tot)])
    
def simulation_no_division(cells,dt,rand=rand):
    while True:
        cells.mesh.move(dr(cells.mesh,dt))
        cells.mesh.update()  
        yield cells

def simulation_with_division(cells,dt,N_steps,rand=rand):
    step = 0.
    while True:
        step += 1
        cells.mesh.move(dr(cells.mesh,dt))
        ready = cells.cell_ids[np.where(cells.properties['lifespan'][cells.cell_ids]<0.0)[0]]
        for mother in ready:
            cells.cell_division(mother,rand)
        cells.update(dt)
        update_progress(step/N_steps)  
        yield cells
        
def run(simulation,N_step,skip):
    return [cells.copy() for cells in itertools.islice(simulation,0,N_step,skip)]

timend = 20.0
timestep = 1.0
centres, ghost_mask = init_centres(4,4,0,0.01,rand)
mesh = Mesh(centres,ghost_mask)
cells = Cells(mesh,rand=rand)
update_progress(0)
history = run(simulation_with_division(cells,dt,timend/dt,rand=rand),timend/dt,timestep/dt)


