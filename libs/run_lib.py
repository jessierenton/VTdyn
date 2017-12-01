import sys
import numpy as np
import itertools
import structure
from structure.mesh import Mesh
from structure.global_constants import *
from structure.cell import Tissue
import structure.initialisation as init

def cycle_function(n,rand,*args):
    return rand.rand(n)*T_G1*2 + T_other

def death_function(n,rand,*args):
    return rand.exponential(T_D,n)

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

        
def simulation_death_and_division(tissue,dt,N_steps,rand):
    step = 0.
    death_time = rand.exponential(1/(len(tissue)*12.))
    while True:
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        ready = np.where(properties['cycle_length']<=tissue.age)[0]
        for mother in ready:
            tissue.add_daughter_cells(mother,rand)
            properties['cycle_length'] = np.append(properties['cycle_length'],cycle_function(2,rand))
            properties['age_of_apoptosis'] = np.append(properties['age_of_apoptosis'],death_function(2,rand))
        tissue.remove(ready)
        tissue.remove(np.where(properties['age_of_apoptosis']<=tissue.age)[0])
        tissue.update(dt)
        update_progress(step/N_steps)  
        yield tissue
        
def run(simulation,N_step,skip):
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

    
def run_simulation_death_and_div(N,timestep,timend,rand):
    tissue = init.init_tissue_torus(N,N,0.01,rand)
    tissue.properties['cycle_length'] = cycle_function(N*N,rand)*rand.rand(N*N)
    tissue.properties['age_of_apoptosis'] = death_function(N*N,rand)
    tissue.age = tissue.properties['cycle_length']*rand.rand(N*N)
    history = run(simulation_death_and_division(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)
    return history
    
