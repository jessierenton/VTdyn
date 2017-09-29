from multiprocessing import Process,Pool,Lock  #parallel processing
import multiprocessing as mp

import sys
import numpy as np
import itertools
import mesh
from mesh import Mesh
from cell import *
from initialisation import *
from plot import *
from data import *

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


def simulation_no_division(tissue,dt,N_steps,rand=rand):
    step = 0
    while True:
        step += 1
        tissue.mesh.move_all(tissue.dr(dt))
        tissue.mesh.update() 
        yield tissue

def simulation_with_division(tissue,dt,N_steps,rand=rand):
    step = 0.
    while True:
        step += 1
        tissue.mesh.move_all(tissue.dr(dt))
        ready = tissue.ready()
        for mother in ready:
            tissue.cell_division(mother,rand)
        tissue.update(dt)
        yield tissue
        
def simulation_death_and_division(tissue,dt,N_steps,rand=rand):
    step = 0.
    while True:
        step += 1
        tissue.mesh.move_all(tissue.dr(dt))
        ready = tissue.ready()
        for mother in ready:
            tissue.cell_division(mother,rand)
        tissue.remove(tissue.dead())
        tissue.update(dt)
        yield tissue
        
def run(simulation,N_step,skip):
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

timend = 200.
timestep = 1.
tissue = init_tissue_torus(20,20,0.01,rand)

tissue = run(simulation_no_division(tissue,dt,1/dt,rand=rand),1/dt,timestep/dt)[-1]
tissue.set_attributes('age',rand.rand(tissue.mesh.N_mesh)*10+1)

history = run(simulation_death_and_division(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)
#
save_N_cell(history,'test')

