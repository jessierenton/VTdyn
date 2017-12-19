import sys
import numpy as np
import itertools
import structure
from structure.global_constants import *
from structure.cell import Tissue
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
        
def run(simulation,N_step,skip):
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------AGE-DEPENDENCE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def cycle_function(n,rand,*args):
    return rand.rand(n)*T_G1*2 + T_other

def death_function(n,rand,*args):
    return rand.exponential(T_D,n)
    
def simulation_age_dependent(tissue,dt,N_steps,rand):
    step = 0.
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
        

def run_simulation_age_dependent(N,timestep,timend,rand):
    tissue = init.init_tissue_torus(N,N,0.01,rand)
    tissue.properties['cycle_length'] = cycle_function(N*N,rand)*rand.rand(N*N)
    tissue.properties['age_of_apoptosis'] = death_function(N*N,rand)
    tissue.age = tissue.properties['cycle_length']*rand.rand(N*N)
    history = run(simulation_age_dependent(tissue,dt,timend/dt,rand=rand),timend/dt,timestep/dt)
    return history
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------SIZE-DEPENDENCE-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def simulation_size_dependent(tissue,dt,N_steps,stepsize,rand):
    step = 0.
    while True:
        # if 1 not in tissue.properties['mutant'] and step%stepsize==1: break
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        ready = np.where(tissue.mesh.areas>=(np.sqrt(3)/2)*(2*RHO)**2)[0]
        for mother in ready:
            tissue.add_daughter_cells(mother,rand)
            # tissue.properties['mutant'] = np.append(tissue.properties['mutant'],[tissue.properties['mutant'][mother]]*2)
            tissue.properties['ancestor'] = np.append(tissue.properties['ancestor'],[tissue.properties['ancestor'][mother]]*2)
        tissue.remove(ready)
        if rand.rand() < (1./T_D)*N*dt:
            tissue.remove(rand.randint(N))
        tissue.update(dt)
        update_progress(step/N_steps)
        yield tissue

def run_simulation_size_dependent(N,timestep,timend,rand):
    ages = rand.rand(N*N)*(T_G1+T_other)
    multiplier = np.mean(RHO+ALPHA*(ages))
    tissue = init.init_tissue_torus_with_multiplier(N,N,0.01,rand,multiplier,ages,save_areas=True)
    tissue.properties['ancestor'] = np.arange(N*N)
    history = run(simulation_size_dependent(tissue,dt,timend/dt,timestep/dt,rand=rand),timend/dt,timestep/dt)
    return history

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------SIZE-DEPENDENCE-WITH-MUTATION---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def simulation_size_dependent_with_mutants(tissue,dt,N_steps,stepsize,rand):
    step = 0.
    while True:
        if (1 not in tissue.properties['mutant'] or 0 not in tissue.properties['mutant']) and step%stepsize==1: break
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr_variableMU(dt,0.1))
        ready = np.where(tissue.mesh.areas>=(np.sqrt(3)/2)*(2*RHO)**2)[0]
        for mother in ready:
            tissue.add_daughter_cells(mother,rand)
            tissue.properties['mutant'] = np.append(tissue.properties['mutant'],[tissue.properties['mutant'][mother]]*2)
        tissue.remove(ready)
        if rand.rand() < (1./T_D)*N*dt:
            tissue.remove(rand.randint(N))
        tissue.update(dt)
        update_progress(step/N_steps)
        yield tissue

def run_simulation_size_dependent_with_mutants(N,timestep,timend,rand):
    ages = rand.rand(N*N)*(T_G1+T_other)
    multiplier = np.mean(RHO+ALPHA*(ages))
    tissue = init.init_tissue_torus_with_multiplier(N,N,0.01,rand,multiplier,ages,save_areas=True)
    tissue.properties['mutant'] = np.zeros(N*N,dtype=int)
    tissue.properties['mutant'][rand.randint(N*N)]=1
    history = run(simulation_size_dependent_with_mutants(tissue,dt,timend/dt,timestep/dt,rand=rand),timend/dt,timestep/dt)
    return history
    
    
