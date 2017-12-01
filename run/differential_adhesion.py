import sys
import numpy as np
import itertools
import structure
from structure.mesh import Mesh
from structure.global_constants import *
from structure.cell import Tissue,Cell
import structure.initialisation as init

rand = np.random.RandomState(42)
N = 20
timend = 10000
timestep=1.

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

def simulation_no_division(tissue,dt,N_steps,rand):
    step = 0
    while True:
        step += 1
        tissue.mesh.move_all(tissue.dr(dt))
        tissue.mesh.update() 
        update_progress(step/N_steps)  
        yield tissue
        
def run(simulation,N_step,skip):
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]


tissue = init.init_tissue(N,N,0.01,rand)
tissue.force_i = tissue.force_adhesion_i
cells_type = np.arange(N**2)%2
rand.shuffle(cells_type)
tissue.set_attributes('type',cells_type)
history = run(simulation_no_division(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)
