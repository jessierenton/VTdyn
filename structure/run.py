import sys
import numpy as np
import itertools
import mesh
from mesh import Mesh
from cell import *
from initialisation import *
from plot import *
from data import *

dt = 0.01 #hours


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


def simulation_no_division(tissue,dt,rand=rand):
    while True:
        tissue.move_all(tissue.dr(dt))
        tissue.mesh.update() 
        yield tissue

def simulation_with_division(tissue,dt,N_steps,rand=rand):
    step = 0.
    while True:
        step += 1
        tissue.move_all(tissue.dr(dt))
        ready = tissue.ready()
        for mother in ready:
            tissue.cell_division(mother,rand)
        tissue.update(dt)
        update_progress(step/N_steps)  
        yield tissue
        
def simulation_death_and_division(tissue,dt,N_steps,rand=rand):
    step = 0.
    while True:
        step += 1
        tissue.mesh.move(dr(tissue.mesh,dt))
        birthready = tissue.mesh.cell_ids[np.where(tissue.properties['lifespan'][tissue.mesh.cell_ids]<0.0)[0]]
        for mother in birthready:
            tissue.cell_division(mother,rand)
        deathready = tissue.mesh.cell_ids[np.where(tissue.properties['deathtime'][tissue.mesh.cell_ids]<0.0)[0]]
        for dead in deathready:
            tissue.removll(dead)
        tissue.update(dt)
        update_progress(step/N_steps)  
        yield tissue
        
def run(simulation,N_step,skip):
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

timend = 2.
timestep = 0.01
tissue = init_tissue(6,6,0.01,rand)
tissue = run(simulation_no_division(tissue,dt,rand=rand),1/dt,timestep/dt)[-1]
tissue.set_attributes('age',rand.rand(tissue.mesh.N_mesh)*11+1)

update_progress(0)
history = run(simulation_with_division(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)



