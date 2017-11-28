import sys
import numpy as np
import itertools
import structure
from structure.mesh import Mesh
from structure.global_constants import *
from structure.cell import Tissue,Cell
import structure.initialisation as init

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

def simulation_with_division(tissue,dt,N_steps,rand):
    step = 0.
    while True:
        step += 1
        tissue.mesh.move_all(tissue.dr(dt))
        ready = tissue.ready()
        for mother in ready:
            tissue.cell_division(mother,rand)
        tissue.update(dt)
        update_progress(step/N_steps)  
        yield tissue
        
def simulation_poisson_death_and_division(tissue,dt,N_steps,rand):
    step = 0.
    death_time = rand.exponential(1/(len(tissue)*12.))
    while True:
        step += 1
        tissue.mesh.move_all(tissue.dr(dt))
        ready = tissue.ready()
        for mother in ready:
            tissue.add_daughter_cells(mother,rand)
        tissue.remove(tissue.dead())
        tissue.update(dt)
        update_progress(step/N_steps)  
        yield tissue
        
def run(simulation,N_step,skip):
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

def run_simulation_no_death(N,timestep,timend,rand):
    tissue = init.init_tissue_torus_agedep(N,N,0.01,rand)
    history = run(simulation_with_division(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)
    return history

def run_simulation_small_removal_and_div(N,timestep,timend,rand):
    tissue = init.init_tissue_torus(20,20,0.01,rand)
    tissue.set_attributes('age',rand.rand(tissue.mesh.N_mesh)*(T_G1+T_other))
    history = run(simulation_small_removal_and_division(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)
    return history
    
def run_simulation_poisson_death_and_div(N,timestep,timend,rand):
    tissue = init.init_tissue_torus(N,N,0.01,rand)
    tissue.set_attributes('age',tissue.by_mesh('cycle_len')*rand.rand(N*N))
    history = run(simulation_poisson_death_and_division(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)
    return history
    
def run_simulation_no_div(N,timestep,timend,rand):
    history = run(simulation_no_division(init.init_tissue_torus(N,N,0.01,rand),dt,timend/dt,rand=rand),timend/dt,timestep/dt)
    return history