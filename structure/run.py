import sys
import numpy as np
import itertools
import mesh
from mesh import Mesh
from cells import *
from functools import partial
from initialisation import *
from plot import *


L0 = 1.0
EPS = 0.05

MU = -50.
ETA = 1.0
dt = 0.005 #hours
r_max = 2.5 #prevents long edges forming in delaunay tri for border cells

rand = np.random.RandomState()

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
        if r_len > r_max: return np.array((0.0,0.0))
        else: return MU*r_hat*(r_len-cells.pref_sep(i,j))

def force_i(mesh,i):
    mapfunc = partial(force_ij,mesh,i)
    return sum(map(mapfunc,mesh.neighbours(i)))
    
def dr(mesh,dt):
    return (dt/ETA)*np.array([force_i(mesh,i) for i in range(0,mesh.N_tot)])


def init_properties(cells,rand=rand):
    N = cells.mesh.N_tot
    cells.properties['sister'] = np.full(N,-1,dtype=int)
    cells.properties['lifespan'] = assign_lifetime(N,rand)
    cells.properties['deathtime'] = assign_deathtime(N,rand)
    cells.properties['age'] = np.zeros(N,dtype=float)
    cells.properties['clones'] = np.array(cells.mesh.cell_ids)

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
        ready = cells.mesh.cell_ids[np.where(cells.properties['lifespan'][cells.mesh.cell_ids]<0.0)[0]]
        for mother in ready:
            cells.cell_division(mother,rand)
        cells.update(dt)
        update_progress(step/N_steps)  
        yield cells
        
def simulation_death_and_division(cells,dt,N_steps,rand=rand):
    step = 0.
    while True:
        step += 1
        cells.mesh.move(dr(cells.mesh,dt))
        birthready = cells.mesh.cell_ids[np.where(cells.properties['lifespan'][cells.mesh.cell_ids]<0.0)[0]]
        for mother in birthready:
            cells.cell_division(mother,rand)
        deathready = cells.mesh.cell_ids[np.where(cells.properties['deathtime'][cells.mesh.cell_ids]<0.0)[0]]
        for dead in deathready:
            cells.cell_apoptosis(dead)
        cells.update(dt)
        update_progress(step/N_steps)  
        yield cells
        
def run(simulation,N_step,skip):
    return [cells.copy() for cells in itertools.islice(simulation,0,N_step,skip)]

timend = 100.
timestep = 0.01
mesh = Mesh(*init_centres(6,6,2,0.001,rand))
cells = Cells(mesh,rand=rand)
init_properties(cells)
cells.properties['type']=np.zeros(cells.mesh.N_tot,dtype=int)
cells.properties['type'][rand.choice(cells.mesh.cell_ids[cells.mesh.ghost_mask])] = 1
update_progress(0)
history = run(simulation_death_and_division(cells,dt,timend/dt,rand=rand),timend/dt,timestep/dt)



