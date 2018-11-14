import os
import sys
import numpy as np
import itertools
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth, MutantSpringForce
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
    step = 0.
    while True:
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        tissue.update(dt)
        update_progress(step/N_steps)  
        yield tissue
        
def run(tissue_original,simulation,N_step,skip):
    return [tissue_original.copy()]+[tissue.copy() for tissue in itertools.islice(simulation,skip-1,N_step,skip)]

def run_generator(simulation,N_step,skip):
    return itertools.islice(simulation,0,N_step,skip)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------POISSON-CONSTANT-POP-SIZE-AND-FITNESS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def simulation_with_mutation_ancestor_tracking(tissue,dt,N_steps,stepsize,rand,mutation_rate,initial=False):
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            r = rand.rand()
            if r < mutation_rate**2: properties['type'] = np.append(properties['type'],rand.randint(0,2,2))
            elif r < mutation_rate: properties['type'] = np.append(properties['type'],[properties['type'][mother],rand.randint(2)])
            else: properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            properties['ancestor'] = np.append(properties['ancestor'],[properties['ancestor'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def simulation_ancestor_tracking(tissue,dt,N_steps,stepsize,rand):
    step = 0.
    while not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            properties['ancestor'] = np.append(properties['ancestor'],[properties['ancestor'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        update_progress(step/N_steps)
        complete = (np.all(tissue.properties['ancestor']==tissue.properties['ancestor'][0]) and step%stepsize==0)
        step += 1 
        yield tissue

def simulation_neutral_with_mutation(tissue,dt,N_steps,stepsize,rand,mutation_rate,initial=False):
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            r = rand.rand()
            if r < mutation_rate**2: properties['type'] = np.append(properties['type'],rand.randint(0,2,2))
            elif r < mutation_rate: properties['type'] = np.append(properties['type'],[properties['type'][mother],rand.randint(2)])
            else: properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def initialise_tissue_ancestors(N,dt,timend,timestep,rand,mutation_rate=None):                
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.properties['ancestor'] = np.arange(N*N)
    tissue.age = np.zeros(N*N,dtype=float)
    if mutation_rate is None: tissue = run(tissue,simulation_ancestor_tracking(tissue,dt,timend/dt,timestep/dt,rand),timend/dt,timestep/dt)[-1]
    else: tissue = run(tissue,simulation_with_mutation_ancestor_tracking(tissue,dt,timend/dt,timestep/dt,rand,mutation_rate),timend/dt,timestep/dt)[-1]
    return tissue

def run_simulation(simulation,N,timestep,timend,rand,mutation_rate,til_fix=True,save_areas=False,tissue=None):
    """prisoners dilemma with decoupled birth and death"""
    if tissue is None:
        tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
        tissue.properties['type'] = np.zeros(N*N,dtype=int)
        tissue.age = np.zeros(N*N,dtype=float)
        tissue = run(tissue, simulation(tissue,dt,10./dt,timestep/dt,rand,DELTA,game,constants,True),10./dt,timestep/dt)[-1]
        tissue.properties['type'][rand.randint(N*N,size=1)]=1
    history = run(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,mutation_rate,~til_fix),timend/dt,timestep/dt)
    return history

